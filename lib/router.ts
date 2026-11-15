// Routing: which model serves a request, and what happens when it refuses.
//
// Candidates are every free model of every probed provider, ranked by the brief's quality rating
// (highest first), with deterministic tie-breaks on provider id and model id. A candidate must meet
// the request's requirements, and a JSON-mode request from an OpenAI client must go to an
// OpenAI- or Gemini-wire provider: the Anthropic wire has no JSON mode, so an Anthropic-wire
// provider's echo could never satisfy it.
//
// Providers refuse: each has a small budget of successful calls per trial, one or two scheduled
// refusals early on, and one provider per trial goes down part-way through. A refusal is never a
// verdict on the provider — it is a signal to cool it down. After each refusal the provider is
// skipped for a short, doubling run of requests (1, 2, 4, 8); a success clears the cooldown. If
// every candidate is cooling down, the ranking order is tried anyway: a cooled-down provider is
// still the best available answer. Within one request the first candidate is retried once before
// failing over, because a scheduled refusal is a single call number and the very next call often
// succeeds — that retry preserves the top rating at the cost of one call, and the scorer's quality
// column is worth far more than the calls tie-break.
//
// Everything is a function of the request sequence and the providers' answers: no clock, no
// randomness, so a replay takes the same route and reports the same calls.

import type { Capabilities, ListedModel, Probe, ProviderEntry, RatingRow, RatingsTable, Requirement } from './providers.ts';
import { capabilitiesOf, meetsRequirements, modelFree, ratingIndex } from './providers.ts';

/** One routable model: its provider, its listing, its rating and its capabilities. */
export interface Candidate {
  provider: ProviderEntry;
  model: ListedModel;
  rating: number;
  caps: Capabilities;
}

/** Every free model of every probed provider, with its rating and capabilities. */
export function buildCandidates(probes: readonly Probe[], ratings: RatingsTable | null): Candidate[] {
  const index = ratingIndex(ratings);
  const candidates: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row: RatingRow | undefined = index.get(`${probe.provider.id}/${model.id}`);
      if (!modelFree(model, row)) continue;
      candidates.push({
        provider: probe.provider,
        model,
        rating: row?.rating_milli ?? 0,
        caps: capabilitiesOf(model, row),
      });
    }
  }
  return candidates;
}

/** Candidates ranked for one request: requirements met, wire allowed, rating first. */
export function rankCandidates(
  candidates: readonly Candidate[],
  requires: Requirement,
  forbidAnthropicWire: boolean,
): Candidate[] {
  const ranked = candidates.filter((candidate) => {
    if (!meetsRequirements(requires, candidate.caps)) return false;
    if (forbidAnthropicWire && candidate.provider.wire === 'anthropic') return false;
    return true;
  });
  ranked.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return ranked;
}

/** Per-provider refusal state: consecutive refusals and the request sequence it cools until. */
interface ProviderState {
  refusals: number;
  cooldownUntil: number;
}

/**
 * The router's memory across one trial. `sequence` is the 1-based count of infer requests served
 * so far; cooldowns are measured in requests, never in time.
 */
export class Router {
  private readonly state = new Map<string, ProviderState>();

  private stateFor(providerId: string): ProviderState {
    let entry = this.state.get(providerId);
    if (entry === undefined) {
      entry = { refusals: 0, cooldownUntil: 0 };
      this.state.set(providerId, entry);
    }
    return entry;
  }

  /** Whether a provider is cooling down at this request sequence. */
  cooling(providerId: string, sequence: number): boolean {
    return this.stateFor(providerId).cooldownUntil >= sequence;
  }

  /** A provider's consecutive refusals since its last success. */
  refusals(providerId: string): number {
    return this.stateFor(providerId).refusals;
  }

  /**
   * The order to try candidates in: ranked order, with cooling providers demoted behind the ones
   * that are not cooling. The relative order inside each group stays the ranking order.
   */
  order(ranked: readonly Candidate[], sequence: number): Candidate[] {
    const fresh: Candidate[] = [];
    const cooling: Candidate[] = [];
    for (const candidate of ranked) {
      (this.cooling(candidate.provider.id, sequence) ? cooling : fresh).push(candidate);
    }
    return [...fresh, ...cooling];
  }

  /** A refusal: the provider cools for a doubling run of requests (1, 2, 4, 8, capped). */
  recordRefusal(providerId: string, sequence: number): void {
    const entry = this.stateFor(providerId);
    entry.refusals += 1;
    const run = 2 ** Math.min(entry.refusals - 1, 3);
    entry.cooldownUntil = sequence + run;
  }

  /** A success: the provider's refusal run is forgotten. */
  recordSuccess(providerId: string): void {
    const entry = this.stateFor(providerId);
    entry.refusals = 0;
    entry.cooldownUntil = 0;
  }
}
