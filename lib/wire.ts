// The wires: the three provider APIs (OpenAI, Anthropic, Gemini) and the two client APIs
// (OpenAI, Anthropic) this server speaks, and the translation between them.
//
// Every request is first read into one canonical shape, then written out in the wire of the
// provider that will serve it, and the provider's answer is read back into one shape and written
// out in the wire the client asked for. The translation is faithful by construction: a provider
// answers with an echo of what it was sent (system text, last user text, tool names, max_tokens,
// stops, image bytes, JSON mode), and the client checks that echo against what it sent, so every
// one of those fields must survive the round trip byte for byte. The one field that cannot is
// JSON mode on the Anthropic wire, which has no JSON mode: the router never sends a JSON-mode
// request to an Anthropic-wire provider.
//
// Providers are always called non-streaming. The client may ask for a stream, in which case this
// server renders the stream itself from the provider's whole answer: the streaming event order is
// then fully in its own hands, and the provider's stream flag is never a check. Usage is passed
// through verbatim, with only the field names translated. No clock and no randomness anywhere:
// ids and timestamps are fixed constants plus the request's sequence number.

import type { Wire, Json } from './providers.ts';
import { isRecord } from './providers.ts';

/** Fixed timestamps: the same on every run, live and replay. */
export const FIXED_CREATED = 1767225600;
export const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/** An image carried by a message: raw bytes (base64) or a URL the provider fetches. */
export type ImageRef =
  | { kind: 'base64'; base64: string; mediaType: string }
  | { kind: 'url'; url: string };

/** A tool call an assistant message carries. `args` is the JSON text of the arguments. */
export interface ToolCall {
  id: string;
  name: string;
  args: string;
}

/** A tool result a tool message carries. */
export interface ToolResult {
  toolCallId: string;
  content: string;
}

/** One message in canonical form. `text` is the message's text parts joined with "\n". */
export interface CanonicalMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: ImageRef[];
  toolCalls: ToolCall[];
  toolResults: ToolResult[];
}

/** One function tool in canonical form. `schema` is its JSON Schema, as the client wrote it. */
export interface CanonicalTool {
  name: string;
  description: string;
  schema: Json;
}

/** A request in canonical form, whichever wire the client spoke. */
export interface CanonicalRequest {
  systemText: string;
  messages: CanonicalMessage[];
  tools: CanonicalTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
  stream: boolean;
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function textOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

/** The text of content: a string as it stands, or its text parts joined with "\n". */
function contentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const part of content) {
    if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') parts.push(part.text);
  }
  return parts.join('\n');
}

/** The images of OpenAI-style content parts. */
function openAiImages(content: unknown): ImageRef[] {
  const images: ImageRef[] = [];
  if (!Array.isArray(content)) return images;
  for (const part of content) {
    if (!isRecord(part) || part.type !== 'image_url' || !isRecord(part.image_url)) continue;
    const url = textOf(part.image_url.url);
    if (url === '') continue;
    const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
    if (match !== null) images.push({ kind: 'base64', base64: match[2], mediaType: match[1] });
    else images.push({ kind: 'url', url });
  }
  return images;
}

/** The images of Anthropic-style content blocks. */
function anthropicImages(content: unknown): ImageRef[] {
  const images: ImageRef[] = [];
  if (!Array.isArray(content)) return images;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
    if (block.source.type === 'base64' && typeof block.source.data === 'string') {
      images.push({ kind: 'base64', base64: block.source.data, mediaType: textOf(block.source.media_type) || 'image/png' });
    } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
      images.push({ kind: 'url', url: block.source.url });
    }
  }
  return images;
}

/** The tool calls of an OpenAI assistant message. */
function openAiToolCalls(message: Json): ToolCall[] {
  const calls: ToolCall[] = [];
  if (!Array.isArray(message.tool_calls)) return calls;
  for (const call of message.tool_calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    calls.push({
      id: textOf(call.id),
      name: textOf(call.function.name),
      args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  return calls;
}

/** The tool calls and results of Anthropic-style content blocks. */
function anthropicToolBlocks(content: unknown): { calls: ToolCall[]; results: ToolResult[] } {
  const calls: ToolCall[] = [];
  const results: ToolResult[] = [];
  if (!Array.isArray(content)) return { calls, results };
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'tool_use') {
      calls.push({ id: textOf(block.id), name: textOf(block.name), args: JSON.stringify(block.input ?? {}) });
    } else if (block.type === 'tool_result') {
      const contentValue = block.content;
      results.push({
        toolCallId: textOf(block.tool_use_id),
        content: typeof contentValue === 'string' ? contentValue : contentText(contentValue),
      });
    }
  }
  return { calls, results };
}

/** An OpenAI request body read into canonical form. */
export function parseOpenAiRequest(body: Json): CanonicalRequest {
  const systemParts: string[] = [];
  const messages: CanonicalMessage[] = [];
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      const role = textOf(raw.role);
      const text = contentText(raw.content);
      if (role === 'system') {
        systemParts.push(text);
      } else if (role === 'assistant') {
        messages.push({ role: 'assistant', text, images: [], toolCalls: openAiToolCalls(raw), toolResults: [] });
      } else if (role === 'tool') {
        messages.push({ role: 'tool', text, images: [], toolCalls: [], toolResults: [{ toolCallId: textOf(raw.tool_call_id), content: text }] });
      } else {
        messages.push({ role: 'user', text, images: openAiImages(raw.content), toolCalls: [], toolResults: [] });
      }
    }
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function)) continue;
      tools.push({
        name: textOf(raw.function.name),
        description: textOf(raw.function.description),
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : {},
      });
    }
  }
  const stop = body.stop;
  const stops = typeof stop === 'string' ? [stop] : Array.isArray(stop) ? stop.filter((s): s is string => typeof s === 'string') : [];
  const format = isRecord(body.response_format) ? textOf(body.response_format.type) : '';
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops,
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stream: body.stream === true,
  };
}

/** An Anthropic request body read into canonical form. */
export function parseAnthropicRequest(body: Json): CanonicalRequest {
  const systemText = contentText(body.system);
  const messages: CanonicalMessage[] = [];
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      const text = contentText(raw.content);
      const { calls, results } = anthropicToolBlocks(raw.content);
      if (textOf(raw.role) === 'assistant') {
        messages.push({ role: 'assistant', text, images: [], toolCalls: calls, toolResults: [] });
      } else if (results.length > 0) {
        messages.push({ role: 'tool', text, images: [], toolCalls: [], toolResults: results });
      } else {
        messages.push({ role: 'user', text, images: anthropicImages(raw.content), toolCalls: [], toolResults: [] });
      }
    }
  }
  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw)) continue;
      tools.push({
        name: textOf(raw.name),
        description: textOf(raw.description),
        schema: isRecord(raw.input_schema) ? raw.input_schema : {},
      });
    }
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  return {
    systemText,
    messages,
    tools,
    maxTokens: numOrNull(body.max_tokens),
    stops,
    // The Anthropic wire has no JSON mode, and its echo always says so.
    jsonMode: false,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stream: body.stream === true,
  };
}

/** An image as an OpenAI image_url part: a data URL for bytes, the URL itself otherwise. */
function openAiImagePart(image: ImageRef): Json {
  return image.kind === 'base64'
    ? { type: 'image_url', image_url: { url: `data:${image.mediaType};base64,${image.base64}` } }
    : { type: 'image_url', image_url: { url: image.url } };
}

/** An image as an Anthropic image block. */
function anthropicImageBlock(image: ImageRef): Json {
  return image.kind === 'base64'
    ? { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.base64 } }
    : { type: 'image', source: { type: 'url', url: image.url } };
}

/** An image as a Gemini part: inline data for bytes, a file reference for a URL. */
function geminiImagePart(image: ImageRef): Json {
  return image.kind === 'base64'
    ? { inlineData: { mimeType: image.mediaType, data: image.base64 } }
    : { fileData: { mimeType: 'image/png', fileUri: image.url } };
}

/** A canonical request written as an OpenAI chat completions body. Never streaming. */
export function buildOpenAiRequest(request: CanonicalRequest, model: string): Json {
  const messages: Json[] = [];
  if (request.systemText !== '') messages.push({ role: 'system', content: request.systemText });
  for (const message of request.messages) {
    if (message.role === 'assistant') {
      const out: Json = { role: 'assistant', content: message.text };
      if (message.toolCalls.length > 0) {
        out.tool_calls = message.toolCalls.map((call) => ({
          id: call.id,
          type: 'function',
          function: { name: call.name, arguments: call.args },
        }));
      }
      messages.push(out);
    } else if (message.role === 'tool') {
      for (const result of message.toolResults) {
        messages.push({ role: 'tool', tool_call_id: result.toolCallId, content: result.content });
      }
      if (message.toolResults.length === 0) messages.push({ role: 'tool', tool_call_id: '', content: message.text });
    } else if (message.images.length > 0) {
      messages.push({
        role: 'user',
        content: [{ type: 'text', text: message.text }, ...message.images.map(openAiImagePart)],
      });
    } else {
      messages.push({ role: 'user', content: message.text });
    }
  }
  const body: Json = { model, messages, stream: false };
  if (request.maxTokens !== null) body.max_tokens = request.maxTokens;
  if (request.temperature !== null) body.temperature = request.temperature;
  if (request.topP !== null) body.top_p = request.topP;
  if (request.stops.length > 0) body.stop = [...request.stops];
  if (request.tools.length > 0) {
    body.tools = request.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.schema },
    }));
  }
  if (request.jsonMode) body.response_format = { type: 'json_object' };
  return body;
}

/** A canonical request written as an Anthropic messages body. Never streaming. */
export function buildAnthropicRequest(request: CanonicalRequest, model: string): Json {
  const body: Json = { model, messages: [], stream: false };
  if (request.systemText !== '') body.system = request.systemText;
  const messages: Json[] = [];
  for (const message of request.messages) {
    if (message.role === 'assistant') {
      if (message.toolCalls.length === 0) {
        messages.push({ role: 'assistant', content: message.text });
      } else {
        const blocks: Json[] = [{ type: 'text', text: message.text }];
        for (const call of message.toolCalls) {
          let input: unknown = {};
          try {
            input = JSON.parse(call.args);
          } catch {
            input = {};
          }
          blocks.push({ type: 'tool_use', id: call.id, name: call.name, input });
        }
        messages.push({ role: 'assistant', content: blocks });
      }
    } else if (message.role === 'tool') {
      const blocks: Json[] = message.toolResults.map((result) => ({
        type: 'tool_result',
        tool_use_id: result.toolCallId,
        content: result.content,
      }));
      if (blocks.length === 0) blocks.push({ type: 'tool_result', tool_use_id: '', content: message.text });
      messages.push({ role: 'user', content: blocks });
    } else if (message.images.length > 0) {
      messages.push({
        role: 'user',
        content: [{ type: 'text', text: message.text }, ...message.images.map(anthropicImageBlock)],
      });
    } else {
      messages.push({ role: 'user', content: message.text });
    }
  }
  body.messages = messages;
  if (request.maxTokens !== null) body.max_tokens = request.maxTokens;
  if (request.temperature !== null) body.temperature = request.temperature;
  if (request.topP !== null) body.top_p = request.topP;
  if (request.stops.length > 0) body.stop_sequences = [...request.stops];
  if (request.tools.length > 0) {
    body.tools = request.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.schema }));
  }
  return body;
}

/** A canonical request written as a Gemini generateContent body. The model goes in the URL. */
export function buildGeminiRequest(request: CanonicalRequest): Json {
  const body: Json = {};
  if (request.systemText !== '') body.systemInstruction = { parts: [{ text: request.systemText }] };
  const contents: Json[] = [];
  for (const message of request.messages) {
    const parts: Json[] = [];
    if (message.text !== '') parts.push({ text: message.text });
    for (const image of message.images) parts.push(geminiImagePart(image));
    for (const call of message.toolCalls) {
      let args: unknown = {};
      try {
        args = JSON.parse(call.args);
      } catch {
        args = {};
      }
      parts.push({ functionCall: { name: call.name, args } });
    }
    for (const result of message.toolResults) {
      parts.push({ functionResponse: { name: result.toolCallId, response: { content: result.content } } });
    }
    if (parts.length === 0) parts.push({ text: '' });
    const role = message.role === 'assistant' ? 'model' : message.role === 'tool' ? 'function' : 'user';
    contents.push({ role, parts });
  }
  body.contents = contents;
  if (request.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: request.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.schema,
        })),
      },
    ];
  }
  const config: Json = {};
  if (request.maxTokens !== null) config.maxOutputTokens = request.maxTokens;
  if (request.stops.length > 0) config.stopSequences = [...request.stops];
  if (request.jsonMode) config.responseMimeType = 'application/json';
  if (request.temperature !== null) config.temperature = request.temperature;
  if (request.topP !== null) config.topP = request.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

/** A canonical request written in a provider's wire. */
export function buildProviderRequest(wire: Wire, request: CanonicalRequest, model: string): Json {
  if (wire === 'anthropic') return buildAnthropicRequest(request, model);
  if (wire === 'gemini') return buildGeminiRequest(request);
  return buildOpenAiRequest(request, model);
}

/**
 * Stop-reason families, shared by all three wires: a response's stop reason counts when it is in
 * the same family as the provider's, whatever the spelling.
 *   0 stop / end_turn / STOP, 1 length / max_tokens / MAX_TOKENS,
 *   2 tool_calls / function_call / tool_use, 3 content_filter / refusal / SAFETY.
 */
export function stopFamily(reason: string): number {
  switch (reason) {
    case 'stop':
    case 'end_turn':
    case 'STOP':
      return 0;
    case 'length':
    case 'max_tokens':
    case 'MAX_TOKENS':
      return 1;
    case 'tool_calls':
    case 'function_call':
    case 'tool_use':
      return 2;
    case 'content_filter':
    case 'refusal':
    case 'SAFETY':
      return 3;
    default:
      return 0;
  }
}

const OPENAI_STOP = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOP = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

/** A provider's answer in canonical form: its text, its usage, its stop family. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopFamily: number;
}

/** The text of an answer's content: a string as it stands, text parts joined with nothing. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const part of content) {
    if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') parts.push(part.text);
  }
  return parts.join('');
}

/** An OpenAI chat completion answer read into canonical form. */
export function parseOpenAiAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.choices) || body.choices.length === 0) return null;
  const choice = body.choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: answerText(choice.message.content),
    inputTokens: numOrNull(usage.prompt_tokens) ?? numOrNull(usage.input_tokens) ?? 0,
    outputTokens: numOrNull(usage.completion_tokens) ?? numOrNull(usage.output_tokens) ?? 0,
    stopFamily: stopFamily(textOf(choice.finish_reason)),
  };
}

/** An Anthropic messages answer read into canonical form. */
export function parseAnthropicAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: answerText(body.content),
    inputTokens: numOrNull(usage.input_tokens) ?? 0,
    outputTokens: numOrNull(usage.output_tokens) ?? 0,
    stopFamily: stopFamily(textOf(body.stop_reason)),
  };
}

/** A Gemini generateContent answer read into canonical form. */
export function parseGeminiAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const candidate = body.candidates[0];
  if (!isRecord(candidate) || !isRecord(candidate.content)) return null;
  const parts = candidate.content.parts;
  const texts: string[] = [];
  if (Array.isArray(parts)) {
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    inputTokens: numOrNull(usage.promptTokenCount) ?? 0,
    outputTokens: numOrNull(usage.candidatesTokenCount) ?? 0,
    stopFamily: stopFamily(textOf(candidate.finishReason)),
  };
}

/** A provider's answer read in its own wire. */
export function parseProviderAnswer(wire: Wire, body: unknown): ProviderAnswer | null {
  if (wire === 'anthropic') return parseAnthropicAnswer(body);
  if (wire === 'gemini') return parseGeminiAnswer(body);
  return parseOpenAiAnswer(body);
}

/** Text split into short pieces for streaming deltas, in order, concatenating to the whole. */
export function chunkText(text: string, size = 13): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += size) chunks.push(text.slice(i, i + size));
  return chunks;
}

function sseData(payload: Json): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function sseEvent(name: string, payload: Json): string {
  return `event: ${name}\ndata: ${JSON.stringify(payload)}\n\n`;
}

/** An OpenAI chat completion: one JSON object, the provider's usage passed through. */
export function renderOpenAiJson(model: string, answer: ProviderAnswer, sequence: number): Json {
  return {
    id: `chatcmpl-req${sequence}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: OPENAI_STOP[answer.stopFamily] ?? 'stop',
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/**
 * An OpenAI chat completion stream: a role chunk, the content deltas, exactly one finish chunk, a
 * usage chunk, and [DONE] last. Nothing follows the finish chunk but the usage chunk and [DONE].
 */
export function renderOpenAiStream(model: string, answer: ProviderAnswer, sequence: number): string {
  const head = { id: `chatcmpl-req${sequence}`, object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  let out = sseData({ ...head, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] });
  for (const piece of chunkText(answer.text)) {
    out += sseData({ ...head, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  out += sseData({ ...head, choices: [{ index: 0, delta: {}, finish_reason: OPENAI_STOP[answer.stopFamily] ?? 'stop' }] });
  out += sseData({
    ...head,
    choices: [],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** An Anthropic message: one JSON object, the provider's usage passed through. */
export function renderAnthropicJson(model: string, answer: ProviderAnswer, sequence: number): Json {
  return {
    id: `msg-req${sequence}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: ANTHROPIC_STOP[answer.stopFamily] ?? 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/**
 * An Anthropic message stream: message_start (carrying the input tokens), one content block opened,
 * deltaed and closed, message_delta (carrying the output tokens), and message_stop last.
 */
export function renderAnthropicStream(model: string, answer: ProviderAnswer, sequence: number): string {
  const id = `msg-req${sequence}`;
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  out += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of chunkText(answer.text)) {
    out += sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: ANTHROPIC_STOP[answer.stopFamily] ?? 'end_turn', stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}

/** An error in the OpenAI wire: a 502 when no provider could serve the request. */
export function renderOpenAiError(message: string): Json {
  return { error: { message, type: 'server_error', code: 'upstream_unavailable' } };
}

/** An error in the Anthropic wire: a 502 when no provider could serve the request. */
export function renderAnthropicError(message: string): Json {
  return { type: 'error', error: { type: 'overloaded_error', message } };
}

/** One model of the served list: its id, the provider that serves it, its display name. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** The OpenAI shape of GET /v1/models: the free ids of the registered catalog, once each. */
export function renderOpenAiModelList(models: ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((model) => ({ id: model.id, object: 'model', created: FIXED_CREATED, owned_by: model.provider })),
  };
}

/** The Anthropic shape of GET /v1/models: the same ids, in the Anthropic wire's spelling. */
export function renderAnthropicModelList(models: ServedModel[]): Json {
  return {
    data: models.map((model) => ({ id: model.id, display_name: model.display_name, created_at: FIXED_CREATED_AT, type: 'model' })),
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
    has_more: false,
  };
}
