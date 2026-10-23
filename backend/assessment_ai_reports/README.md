# Assessment AI Report Service

This module owns only the OpenAI call used to generate assessment-report narratives.
The Scorguide application remains responsible for collecting assessment data, calculating
scores, building the complete prompt, normalizing the returned sections, and caching the
final report.

## Python API

```python
from backend.assessment_ai_reports import generate_assessment_ai_report

report = generate_assessment_ai_report(
    prompt=prepared_prompt,
    assessment_type="sc_maturity_assessment",
)
```

## HTTP API

`POST /api/assessment-reports/generate`

```json
{
  "assessment_type": "sc_maturity_assessment",
  "prompt": "The complete assessment report prompt"
}
```

Supported assessment types are `sc_maturity_assessment`,
`digital_readiness_assessment`, and `sop_assessment`.

The endpoint responds with:

```json
{
  "assessment_type": "sc_maturity_assessment",
  "model": "configured model name",
  "report": {
    "as_is_state": [],
    "consulting_insights": [],
    "team_function_alignment_insights": []
  }
}
```

Required environment settings:

- `OPENAI_API_KEY`
- `ASSESSMENT_AI_REPORT_MODEL`

Optional setting:

- `ASSESSMENT_AI_REPORT_MAX_PROMPT_CHARS` (defaults to `500000`)

Prompts and generated reports are not written to local files by this service.
