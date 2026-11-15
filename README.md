# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/zevabagus-spec/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/zevabagus-spec/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A single server speaking both the OpenAI and Anthropic APIs, routing each request over ten simulated free-tier model providers on three wires (OpenAI, Anthropic, Gemini). Requests spread across all ten providers under per-provider repeat caps — a provider's first call is its safest, and every refusal costs a whole extra call — with a deficit guard that holds the weighted rating gap inside a budget keeping quality at 98 or better, and failover to the next-best untried provider when one refuses. Every answer is rendered in the wire the client asked for — JSON or a well-ordered stream — with the provider's usage passed through verbatim. Deterministic throughout: no clock, no randomness, no model call of its own. Node 22 standard library only, no dependencies.

## Run it

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=prompt.txt node main.ts
```

`node main.ts --list` prints the catalog report the server would register, without running a trial.

---

<sub>Entered on [make.fun](https://make.fun/?ref=pkrtjm47&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
