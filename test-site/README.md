# ConvRot test site

Minimal, standalone test harness for the Kobold Speed Lab RunPod endpoint.
No Supabase, no Railway, no Vercel, no auth - a plain Express server plus
one HTML page. It exists purely to submit generations and time them, kept
fully independent of the production site's infrastructure.

## Setup

```
cd test-site
npm install
cp .env.example .env
# fill in RUNPOD_API_KEY, RUNPOD_S3_ENDPOINT, RUNPOD_S3_ACCESS_KEY,
# RUNPOD_S3_SECRET_KEY (see your RunPod dashboard, or ask for the values)
npm start
```

Then open http://localhost:3000

## What it does

- `POST /api/generate` - submits a classic (non-session) job straight to
  the RunPod endpoint in `RUNPOD_ENDPOINT_ID` (defaults to Kobold Speed
  Lab). No `session_id` is sent, so the worker's handler.py never touches
  Supabase - it goes straight through `run_generation()`.
- `GET /api/status/:jobId` - proxies RunPod's own status endpoint so the
  page can poll without exposing the API key to the browser.
- `GET /api/video/:key` - once the job completes, fetches the finished
  video from the endpoint's S3-compatible network volume and streams it
  back for preview/download.

## Note on the ConvRot build

Kobold Speed Lab is currently running the stock `minimax-h3-worker`
Docker image - the ConvRot-patched `koboldcpp_cublas.so` built and
verified on the GPU pod has not been deployed into any worker image yet.
This site works against the stock build today (useful for a baseline and
for proving the pipeline itself works). Getting ConvRot into an actual
generation requires rebuilding the worker's Docker image with the patched
binary and pushing it to the registry this endpoint pulls from - a
separate follow-up step.
