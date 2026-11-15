// free-model-aggregator: one server, two client APIs, ten simulated free-tier providers.
//
// The server reads its brief (provider directory, quality table, workload manifest), probes every
// provider's catalog route, and registers the resulting catalog with the trial environment. It
// then serves the client's twenty requests, one at a time, spreading them
// across all ten providers under per-provider repeat caps: a provider's
// first call is its safest, and every refusal costs a whole extra call, so
// the router prefers untried providers and lower-rated models where the
// brief's quality table leaves room, holding the weighted rating deficit
// inside a budget that keeps quality at 98 or better. It renders each
// answer in the wire the client spoke — JSON or a well-ordered stream, with
// the provider's usage passed through verbatim. When the environment
// reports the workload done, the server exits 0.
//
// Usage: node main.ts [--list]
//   --list prints the catalog report (the document registration would send) and exits.
//   Without it the server runs a trial against $MAKEFUN_ENV_URL and $MAKEFUN_PROMPT_FILE.
//
// The server is deterministic: no clock, no randomness, no model call of its own. Every choice is
// a function of the brief, the probe answers, the requests, and the providers' answers, in that
// order, so a replay reproduces the same catalog, the same routes and the same calls.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';

import {
  authHeaders,
  buildCatalog,
  joinBase,
  modelFree,
  probeProvider,
  ratingIndex,
  readBrief,
  type Brief,
  type CatalogReport,
  type Probe,
  type RatingRow,
  type RatingsTable,
  type Requirement,
  type WorkloadRequest,
} from './lib/providers.ts';
import { isRecord, type Json } from './lib/providers.ts';
import { buildCandidates, Router, type Candidate } from './lib/router.ts';
import {
  buildProviderRequest,
  parseAnthropicRequest,
  parseOpenAiRequest,
  parseProviderAnswer,
  renderAnthropicError,
  renderAnthropicJson,
  renderAnthropicModelList,
  renderAnthropicStream,
  renderOpenAiError,
  renderOpenAiJson,
  renderOpenAiModelList,
  renderOpenAiStream,
  type CanonicalRequest,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const PROVIDER_TIMEOUT_MS = 5000;
const PROBE_ATTEMPTS = 3;
const MAX_ATTEMPTS_PER_REQUEST = 8;

interface Loaded {
  brief: Brief;
  probes: Probe[];
  catalog: CatalogReport;
  candidates: Candidate[];
  servedModels: ServedModel[];
}

/** Probe every provider, retrying a failed probe, and build the catalog and candidates. */
async function load(envUrl: string, promptFile: string): Promise<Loaded> {
  const brief = readBrief(promptFile);
  if (brief.error !== null) throw new Error(brief.error);
  const probes = await Promise.all(
    brief.providers.map(async (provider) => {
      let probe: Probe | null = null;
      for (let attempt = 0; attempt < PROBE_ATTEMPTS; attempt++) {
        probe = await probeProvider(envUrl, provider);
        if (probe.ok) break;
      }
      return probe as Probe;
    }),
  );
  const catalog = buildCatalog(probes, brief.ratings);
  const candidates = buildCandidates(probes, brief.ratings);
  const servedModels = servedModelList(probes, brief.ratings);
  return { brief, probes, catalog, candidates, servedModels };
}

/**
 * The free models the server lists on GET /v1/models: one row per free id, in catalog order
 * (providers in directory order, models sorted by id), the first provider winning a shared id.
 */
function servedModelList(probes: readonly Probe[], ratings: RatingsTable | null): ServedModel[] {
  const index = ratingIndex(ratings);
  const seen = new Set<string>();
  const models: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      const row: RatingRow | undefined = index.get(`${probe.provider.id}/${model.id}`);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      models.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return models;
}

/** The requirements of one request: its manifest row where the manifest has one, the body otherwise. */
function requirementsFor(
  manifest: readonly WorkloadRequest[],
  index: number,
  body: Json,
  api: 'openai' | 'anthropic',
): Requirement {
  const row = manifest[index];
  if (row !== undefined) return row.requires;
  return requirementsFromBody(body, api);
}

function bodyHasImage(body: Json): boolean {
  if (!Array.isArray(body.messages)) return false;
  for (const raw of body.messages) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const part of raw.content) {
      if (isRecord(part) && (part.type === 'image_url' || part.type === 'image')) return true;
    }
  }
  return false;
}

function requirementsFromBody(body: Json, api: 'openai' | 'anthropic'): Requirement {
  const requires: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  if (bodyHasImage(body)) requires.vision = true;
  if (api === 'openai' && isRecord(body.response_format)) {
    const type = body.response_format.type;
    if (type === 'json_object' || type === 'json_schema') requires.json = true;
  }
  return requires;
}

type ProviderResult = { ok: true; answer: ProviderAnswer } | { ok: false; status: number };

/** One inference call to one provider, in its own wire, non-streaming. */
async function callProvider(
  envUrl: string,
  candidate: Candidate,
  request: CanonicalRequest,
): Promise<ProviderResult> {
  const { provider, model } = candidate;
  const base = joinBase(envUrl, provider.base);
  let url: string;
  if (provider.wire === 'gemini') {
    url = `${base}/v1beta/models/${encodeURIComponent(model.id)}:generateContent`;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
  } else {
    url = `${base}/v1/chat/completions`;
  }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(buildProviderRequest(provider.wire, request, model.id)),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
    if (!res.ok) return { ok: false, status: res.status };
    const answer = parseProviderAnswer(provider.wire, (await res.json()) as unknown);
    if (answer === null) return { ok: false, status: 502 };
    return { ok: true, answer };
  } catch {
    return { ok: false, status: 0 };
  }
}

/**
 * Serve one request: ask the router for one candidate at a time and stop at
 * the first success. A refusal is never retried on the same provider while
 * another eligible provider is untried — the router's order already skips
 * it — so one refusal costs exactly one extra provider call. A 400 or 401
 * (a wrong key, an unknown or paid model) will not right itself on the next
 * call to that provider, and the tried-set keeps the request moving on to
 * the next provider rather than spending another call there.
 */
async function serve(
  envUrl: string,
  router: Router,
  request: CanonicalRequest,
  requires: Requirement,
  forbidAnthropicWire: boolean,
  sequence: number,
): Promise<{ model: string; answer: ProviderAnswer } | null> {
  const tried = new Set<string>();
  for (let attempts = 0; attempts < MAX_ATTEMPTS_PER_REQUEST; attempts++) {
    const candidate = router.pick(requires, forbidAnthropicWire, sequence, tried);
    if (candidate === null) break;
    tried.add(candidate.provider.id);
    const result = await callProvider(envUrl, candidate, request);
    router.recordAttempt(candidate.provider.id);
    if (result.ok) {
      router.recordSuccess(candidate.provider.id, sequence, candidate.rating);
      return { model: candidate.model.id, answer: result.answer };
    }
    router.recordRefusal(candidate.provider.id);
  }
  return null;
}

/** A request body read as JSON, capped, or null when it is not JSON. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: Json): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

async function main(): Promise<void> {
  const listMode = process.argv.includes('--list');
  const envUrl = process.env.MAKEFUN_ENV_URL ?? '';
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? '';

  if (listMode) {
    try {
      const loaded = await load(envUrl, promptFile);
      process.stdout.write(`${JSON.stringify(loaded.catalog)}\n`);
    } catch {
      process.stdout.write('{"providers":[]}\n');
    }
    return;
  }

  if (envUrl === '') {
    console.error('MAKEFUN_ENV_URL is not set');
    process.exitCode = 1;
    return;
  }
  const loaded = await load(envUrl, promptFile);
  const manifest = loaded.brief.workload?.manifest ?? [];
  const best = new Map<string, number>(
    (loaded.brief.ratings?.best ?? []).map((row) => [row.id, row.rating_milli]),
  );
  const router = new Router(best, manifest, loaded.candidates);
  let inferCount = 0;

  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://localhost');
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        // The client warms up with both auth headers: an x-api-key call expects the Anthropic
        // shape, anything else the OpenAI shape. Both list the same free ids.
        if (typeof req.headers['x-api-key'] === 'string') {
          sendJson(res, 200, renderAnthropicModelList(loaded.servedModels));
        } else {
          sendJson(res, 200, renderOpenAiModelList(loaded.servedModels));
        }
        return;
      }
      const isOpenAi = req.method === 'POST' && url.pathname === '/v1/chat/completions';
      const isAnthropic = req.method === 'POST' && url.pathname === '/v1/messages';
      if (!isOpenAi && !isAnthropic) {
        sendJson(res, 404, { error: 'not found' });
        return;
      }
      const body = await readJson(req);
      if (body === null) {
        sendJson(res, 400, isOpenAi ? renderOpenAiError('request body must be a JSON object') : renderAnthropicError('request body must be a JSON object'));
        return;
      }
      const api: 'openai' | 'anthropic' = isOpenAi ? 'openai' : 'anthropic';
      const request = isOpenAi ? parseOpenAiRequest(body) : parseAnthropicRequest(body);
      const sequence = inferCount + 1;
      inferCount += 1;
      const requires = requirementsFor(manifest, sequence - 1, body, api);
      // JSON mode cannot survive the Anthropic wire: keep JSON-mode OpenAI requests off it.
      const served = await serve(envUrl, router, request, requires, isOpenAi && request.jsonMode, sequence);
      if (served === null) {
        sendJson(res, 502, isOpenAi ? renderOpenAiError('no provider could serve the request') : renderAnthropicError('no provider could serve the request'));
        return;
      }
      if (request.stream) {
        sendStream(res, isOpenAi ? renderOpenAiStream(served.model, served.answer, sequence) : renderAnthropicStream(served.model, served.answer, sequence));
      } else {
        sendJson(res, 200, isOpenAi ? renderOpenAiJson(served.model, served.answer, sequence) : renderAnthropicJson(served.model, served.answer, sequence));
      }
    } catch (err) {
      console.error(`request failed: ${(err as Error).message}`);
      if (!res.headersSent) sendJson(res, 500, { error: 'internal error' });
    }
  });

  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  if (address === null || typeof address === 'string') {
    console.error('could not bind a port');
    process.exitCode = 1;
    return;
  }
  const port = address.port;

  const register = await fetch(`${envUrl.replace(/\/+$/, '')}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: loaded.catalog }),
  });
  if (!register.ok) {
    console.error(`register answered HTTP ${register.status}`);
    process.exitCode = 1;
    return;
  }
  const registered: unknown = await register.json();
  if (!isRecord(registered) || registered.ok !== true) {
    console.error('register did not confirm');
    process.exitCode = 1;
    return;
  }

  // The workload runs while this server polls; the moment the environment says done, exit 0.
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`${envUrl.replace(/\/+$/, '')}/work`, { signal: AbortSignal.timeout(5000) });
      const work: unknown = await res.json();
      if (isRecord(work) && work.done === true) {
        clearInterval(timer);
        server.close();
        process.exit(0);
      }
    } catch {
      // A missed poll changes nothing; the next one follows in 100 ms.
    }
  }, POLL_MS);
  timer.unref();

  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
}

main().catch((err: unknown) => {
  console.error((err as Error).message ?? err);
  process.exit(1);
});
