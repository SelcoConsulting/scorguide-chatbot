# SCOR Assistant deployment package

Deploy this folder itself as the Azure App Service application root.

Azure App Service configuration:

- Runtime: Linux / Python
- Startup command: `python app.py`
- App setting: `SCM_DO_BUILD_DURING_DEPLOYMENT=1`
- Required app settings: `MILVUS_URI`, `MILVUS_COLLECTION`, `AI_SERVICE_SECRET_KEY`
- Add `MILVUS_TOKEN` and `MILVUS_DATABASE` when required by the Milvus deployment.
- Add `CORS_ALLOWED_ORIGINS` when a browser frontend calls this service directly.

After deployment, verify `/health` and `/ready`.

Requests to `/api/chat`, `/api/messages`, and
`/api/assessment-reports/generate` must include the shared service secret:

`X-AI-Service-Key: <AI_SERVICE_SECRET_KEY>`

This package intentionally excludes `.env`, virtual environments, Git metadata,
caches, logs, and local build scripts.

Set the environment variable `MODEL_PROVIDER` to `"openai"` or `"azure"`
(default: `"openai"`). Local development can also set it in `.env`; existing
environment variables take precedence.

- OpenAI requires `OPENAI_API_KEY` in the environment or `.env`.
- Azure requires `AZURE_OPENAI_KEY` and `AZURE_OPENAI_ENDPOINT`.
- Model names remain in `backend/config.py`; on Azure they must match deployment names.
- Restart the service after changing environment variables; no redeployment is needed.
