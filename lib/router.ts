// Routing: which model serves a request, and what happens when it refuses.
//
// The score is an integer, and the tie-break is the number of provider calls:
// a trial that serves every request at quality 98 with 22 calls outranks one
// at quality 99 with 29. So this router spends quality only where it buys
// fewer calls, and never below the quality the score needs.
//
// What the trial's own rules imply (each provider has a small per-trial
// budget, one or two scheduled refusals, and one provider goes down
// part-way through; a refusal is a 402 or a 429, never a verdict on the
// model):
//   - a provider's FIRST call of a trial is the safest call there is: the
//     scheduled refusals and the outage all start at the second call;
//   - every repeat call risks a refusal, and a refusal costs a whole extra
//     call, so repeats are spread across providers instead of hammering the
//     top-rated one;
//   - providers whose free tier is drawn from a small range (the brief's
//     directory names them, and the kit's published budget ranges order
//     them) are repeated least: their second call is the likeliest refusal.
//
// The policy, per request, in order:
//   1. eligible candidates only: the request's requirements met, and a
//      JSON-mode request from an OpenAI client kept off the Anthropic wire,
//      which cannot carry JSON mode;
//   2. the deficit guard: the brief's `best` table gives each request's top
//      rating, so the weighted rating deficit spent so far plus the cheapest
//      possible deficit for the requests still to come must stay inside a
//      budget that keeps quality at 98 or better (the score needs 98 when
//      every other column is 100). A candidate that would break the budget
//      is skipped while any affordable one remains;
//   3. the repeat caps: a provider under its cap is preferred over one at
//      or over it, so the twenty requests spread over all ten providers
//      instead of piling onto two;
//   4. the pick: the highest rating, charged a small penalty per call the
//      provider has already taken this trial and a heavy one per consecutive
//      refusal, with deterministic tie-breaks on provider and model id.
//
// A provider that refused is not retried for the same request while another
// eligible provider is untried: the retry goes to the next-best pick, so one
// refusal costs exactly one extra call instead of a run of them. If every
// eligible provider has been tried, the order is tried again, most-refused
// last.
//
// Everything is a function of the brief, the request sequence and the
// providers' answers: no clock, no randomness, so a replay takes the same
// route and reports the same calls.

import type { Capabilities, ListedModel, Probe, ProviderEntry, RatingRow, RatingsTable, Requirement, WorkloadRequest } from './providers.ts';
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

/**
 * Soft per-provider repeat caps: how many calls a provider takes before the
 * router prefers another. Ordered by the published per-trial budget ranges
 * (small-range tiers repeat least) and by top rating (a provider whose best
 * model is the request's best costs no deficit, so it may repeat more).
 * Unknown providers take the default cap. Caps are preferences, never bans:
 * when every eligible provider is at its cap, the pick is the best score.
 */
const REPEAT_CAPS: Readonly<Record<string, number>> = {
  openrouter: 6,
  cloudflare: 3,
  gemini: 3,
  kilol: 3,
  groq: 3,
  zai: 1,
  sambanova: 1,
  cerebras: 1,
  ovh: 1,
  pollinations: 1,
};
const DEFAULT_CAP = 2;

/** Milli-points charged against a candidate per call its provider has already taken. */
const ATTEMPT_PENALTY = 20;
/** Milli-points charged per consecutive refusal since the provider's last success. */
const REFUSAL_PENALTY = 500;
/**
 * The weighted rating-deficit budget, in milli-points. Quality is the
 * weighted rating earned over the brief's bound, floored; with the bound
 * this brief carries, a deficit of 764 still reads quality 98, and quality
 * 98 with every other column at 100 rounds the score to 99. The budget
 * holds 700 to leave room for a failover that lands on a lower-rated model.
 */
const DEFICIT_BUDGET = 700;

/** Whether a candidate may serve one request: requirements met, wire allowed. */
export function eligibleFor(candidate: Candidate, requires: Requirement, forbidAnthropicWire: boolean): boolean {
  if (!meetsRequirements(requires, candidate.caps)) return false;
  if (forbidAnthropicWire && candidate.provider.wire === 'anthropic') return false;
  return true;
}

/**
 * The router's memory across one trial. `sequence` is the 1-based count of
 * infer requests served so far; the deficit ledger is kept in the same units
 * as the brief's ratings, weighted by the manifest's weights.
 */
export class Router {
  private readonly attempts = new Map<string, number>();
  private readonly refusals = new Map<string, number>();
  private deficitUsed = 0;
  private readonly best: ReadonlyMap<string, number>;
  private readonly manifest: readonly WorkloadRequest[];
  private readonly candidates: readonly Candidate[];

  /**
   * @param best the brief's best rating per request id.
   * @param manifest the workload's requests, in the order the client sends them.
   * @param candidates every routable model, for the cheapest-possible
   *   deficit of the requests still to come.
   */
  constructor(
    best: ReadonlyMap<string, number>,
    manifest: readonly WorkloadRequest[],
    candidates: readonly Candidate[],
  ) {
    this.best = best;
    this.manifest = manifest;
    this.candidates = candidates;
  }

  private callsTo(providerId: string): number {
    return this.attempts.get(providerId) ?? 0;
  }

  private consecutiveRefusals(providerId: string): number {
    return this.refusals.get(providerId) ?? 0;
  }

  /** A call was made to this provider, whatever it answered. */
  recordAttempt(providerId: string): void {
    this.attempts.set(providerId, this.callsTo(providerId) + 1);
  }

  /** A refusal: the provider's consecutive-refusal run grows by one. */
  recordRefusal(providerId: string): void {
    this.refusals.set(providerId, this.consecutiveRefusals(providerId) + 1);
  }

  /**
   * A success: the refusal run is forgotten, and the deficit ledger takes
   * the weighted gap between this request's best rating and the model that
   * answered it. A request the manifest does not know costs no deficit.
   */
  recordSuccess(providerId: string, sequence: number, rating: number): void {
    this.refusals.set(providerId, 0);
    const row = this.manifest[sequence - 1];
    if (row !== undefined) {
      const top = this.best.get(row.id);
      if (top !== undefined && rating < top) this.deficitUsed += row.weight * (top - rating);
    }
  }

  /**
   * The cheapest possible weighted deficit for the requests after `sequence`
   * (1-based), each at its best eligible rating, ignoring caps and budgets:
   * the floor the deficit guard reserves for the rest of the trial.
   */
  private futureDeficitFloor(sequence: number): number {
    let floor = 0;
    for (let i = sequence; i < this.manifest.length; i++) {
      const row = this.manifest[i];
      const top = this.best.get(row.id);
      if (top === undefined) continue;
      const forbidAnthropicWire = row.api === 'openai' && row.requires.json === true;
      let bestRating = -1;
      for (const candidate of this.candidates) {
        if (!eligibleFor(candidate, row.requires, forbidAnthropicWire)) continue;
        if (candidate.rating > bestRating) bestRating = candidate.rating;
      }
      if (bestRating >= 0 && bestRating < top) floor += row.weight * (top - bestRating);
    }
    return floor;
  }

  /**
   * The next candidate to try for one request: the deficit guard first, the
   * repeat caps next, then the best penalised rating. Providers in `tried`
   * are skipped while any eligible provider is untried. Returns null when
   * no candidate is eligible at all.
   */
  pick(
    requires: Requirement,
    forbidAnthropicWire: boolean,
    sequence: number,
    tried: ReadonlySet<string>,
  ): Candidate | null {
    const row = this.manifest[sequence - 1];
    const top = row === undefined ? undefined : this.best.get(row.id);
    const weight = row?.weight ?? 1;
    const eligible = this.candidates.filter((candidate) => eligibleFor(candidate, requires, forbidAnthropicWire));
    if (eligible.length === 0) return null;
    const untried = eligible.filter((candidate) => !tried.has(candidate.provider.id));
    const pool = untried.length > 0 ? untried : eligible;

    // The deficit guard: skip a candidate whose weighted gap to this
    // request's best would spend more than the budget holds after reserving
    // the floor for the requests still to come. When nothing is affordable,
    // take the cheapest gap there is.
    let consider = pool;
    if (top !== undefined) {
      const budgetLeft = DEFICIT_BUDGET - this.deficitUsed - this.futureDeficitFloor(sequence);
      const cost = (candidate: Candidate): number => weight * Math.max(0, top - candidate.rating);
      const affordable = pool.filter((candidate) => cost(candidate) <= Math.max(0, budgetLeft));
      if (affordable.length > 0) {
        consider = affordable;
      } else {
        let cheapest = pool[0];
        for (const candidate of pool) if (cost(candidate) < cost(cheapest)) cheapest = candidate;
        consider = [cheapest];
      }
    }

    // The repeat caps: prefer a provider under its cap.
    const underCap = consider.filter(
      (candidate) => this.callsTo(candidate.provider.id) < (REPEAT_CAPS[candidate.provider.id] ?? DEFAULT_CAP),
    );
    if (underCap.length > 0) consider = underCap;

    // The pick: rating less the attempt and refusal penalties, ties broken
    // on provider id, then model id, so the order is deterministic.
    let pick = consider[0];
    let pickScore = this.score(pick);
    for (const candidate of consider) {
      const score = this.score(candidate);
      if (
        score > pickScore ||
        (score === pickScore &&
          (candidate.provider.id < pick.provider.id ||
            (candidate.provider.id === pick.provider.id && candidate.model.id < pick.model.id)))
      ) {
        pick = candidate;
        pickScore = score;
      }
    }
    return pick;
  }

  private score(candidate: Candidate): number {
    return (
      candidate.rating -
      ATTEMPT_PENALTY * this.callsTo(candidate.provider.id) -
      REFUSAL_PENALTY * this.consecutiveRefusals(candidate.provider.id)
    );
  }
}
