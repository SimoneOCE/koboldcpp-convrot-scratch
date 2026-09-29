// Minimal standalone test harness for a RunPod koboldcpp/MiniMax-H3
// serverless endpoint. No Supabase, no Railway, no Vercel, no auth - just
// enough server to hold the RunPod API key (it can't live in the browser)
// and proxy three calls: submit a classic (non-session) generation job,
// poll its status, and stream the finished video back from the endpoint's
// S3-compatible network volume.
require("dotenv").config();

const express = require("express");
const fetch = require("node-fetch");
const path = require("path");
const { S3Client, GetObjectCommand } = require("@aws-sdk/client-s3");

const {
  RUNPOD_API_KEY,
  RUNPOD_ENDPOINT_ID,
  RUNPOD_S3_ENDPOINT,
  RUNPOD_S3_ACCESS_KEY,
  RUNPOD_S3_SECRET_KEY,
  RUNPOD_VOLUME_ID,
  PORT = 3000,
} = process.env;

for (const [name, value] of Object.entries({
  RUNPOD_API_KEY,
  RUNPOD_ENDPOINT_ID,
  RUNPOD_S3_ENDPOINT,
  RUNPOD_S3_ACCESS_KEY,
  RUNPOD_S3_SECRET_KEY,
  RUNPOD_VOLUME_ID,
})) {
  if (!value) {
    console.error(`Missing required env var ${name} - copy .env.example to .env and fill it in.`);
    process.exit(1);
  }
}

const RUNPOD_BASE = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}`;

const s3 = new S3Client({
  endpoint: RUNPOD_S3_ENDPOINT,
  region: "eur-no-1",
  credentials: {
    accessKeyId: RUNPOD_S3_ACCESS_KEY,
    secretAccessKey: RUNPOD_S3_SECRET_KEY,
  },
  forcePathStyle: true,
});

// Same clamps as the production site's server.js - keeps a stray request
// from this test harness from accidentally requesting something huge.
const LIMITS = {
  frames: { min: 1, max: 200 },
  fps: { min: 1, max: 30 },
  steps: { min: 1, max: 40 },
  width: { min: 64, max: 1920 },
  height: { min: 64, max: 1920 },
};

function clamp(input) {
  const out = { ...input };
  for (const [key, { min, max }] of Object.entries(LIMITS)) {
    if (typeof out[key] === "number" && Number.isFinite(out[key])) {
      out[key] = Math.min(max, Math.max(min, out[key]));
    }
  }
  return out;
}

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.post("/api/generate", async (req, res) => {
  try {
    const input = clamp(req.body || {});
    if (!input.prompt) {
      return res.status(400).json({ error: "prompt is required" });
    }

    const runResponse = await fetch(`${RUNPOD_BASE}/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${RUNPOD_API_KEY}`,
      },
      body: JSON.stringify({ input }),
    });

    if (!runResponse.ok) {
      const errText = await runResponse.text();
      throw new Error(`RunPod /run failed: ${runResponse.status} ${errText}`);
    }

    const runData = await runResponse.json();
    if (!runData.id) {
      throw new Error("RunPod did not return a job ID: " + JSON.stringify(runData));
    }

    res.json({ jobId: runData.id });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/status/:jobId", async (req, res) => {
  try {
    const statusResponse = await fetch(
      `${RUNPOD_BASE}/status/${req.params.jobId}`,
      { headers: { Authorization: `Bearer ${RUNPOD_API_KEY}` } },
    );
    const data = await statusResponse.json();
    res.json(data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const VALID_VIDEO_KEY = /^[a-f0-9-]{36}\.(gif|mp4|png|webp|avi)$/i;

app.get("/api/video/:key", async (req, res) => {
  const key = req.params.key;
  if (!VALID_VIDEO_KEY.test(key)) {
    return res.status(400).json({ error: "Invalid key format" });
  }

  try {
    const command = new GetObjectCommand({
      Bucket: RUNPOD_VOLUME_ID,
      Key: `outputs/${key}`,
    });
    const s3Response = await s3.send(command);
    res.setHeader("Content-Type", s3Response.ContentType || "video/mp4");
    res.setHeader("Content-Disposition", `attachment; filename="${key}"`);
    s3Response.Body.pipe(res);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`ConvRot test site running at http://localhost:${PORT}`);
  console.log(`Hitting RunPod endpoint: ${RUNPOD_ENDPOINT_ID}`);
});
