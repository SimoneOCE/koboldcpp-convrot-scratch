// One-off script: frees space on the shared network volume (9tjzvfdfju)
// so the 31.7GB minimax_h3_fl2va_int8_convrot.safetensors download can
// finish - it hit "Disk quota exceeded" partway through because the
// stock model + the failed partial download together left no room.
//
// Deletes:
//   - DasiwaMinimaxH3_dasiwaHybridV1_Q5_0.gguf (13.88GB) - the stock
//     diffusion model. Only Kobold Speed Lab still needs this; it will
//     just re-download it (one-time cost) the next time that endpoint
//     runs a job.
//   - minimax_h3_fl2va_int8_convrot.safetensors + its .aria2 control
//     files - the failed, incomplete download from the last attempt.
//
// Does NOT touch the CLIP model or VAEs - both configs (stock and
// ConvRot) share those, so deleting them would just force a wasteful
// re-download for no benefit.
//
// Usage:
//   node free-volume-space.js
require("dotenv").config();
const { S3Client, DeleteObjectCommand, HeadObjectCommand } = require("@aws-sdk/client-s3");

const {
  RUNPOD_S3_ENDPOINT,
  RUNPOD_S3_ACCESS_KEY,
  RUNPOD_S3_SECRET_KEY,
  RUNPOD_VOLUME_ID,
} = process.env;

for (const [name, value] of Object.entries({
  RUNPOD_S3_ENDPOINT,
  RUNPOD_S3_ACCESS_KEY,
  RUNPOD_S3_SECRET_KEY,
  RUNPOD_VOLUME_ID,
})) {
  if (!value) {
    console.error(`Missing ${name} - fill in .env first (see .env.example).`);
    process.exit(1);
  }
}

const s3 = new S3Client({
  endpoint: RUNPOD_S3_ENDPOINT,
  region: "eur-no-1",
  credentials: {
    accessKeyId: RUNPOD_S3_ACCESS_KEY,
    secretAccessKey: RUNPOD_S3_SECRET_KEY,
  },
  forcePathStyle: true,
});

const KEYS_TO_DELETE = [
  "DasiwaMinimaxH3_dasiwaHybridV1_Q5_0.gguf",
  "minimax_h3_fl2va_int8_convrot.safetensors",
  "minimax_h3_fl2va_int8_convrot.safetensors.aria2",
  "minimax_h3_fl2va_int8_convrot.safetensors.aria2__temp",
];

async function main() {
  for (const key of KEYS_TO_DELETE) {
    try {
      const head = await s3.send(new HeadObjectCommand({ Bucket: RUNPOD_VOLUME_ID, Key: key }));
      console.log(`Deleting ${key} (${(head.ContentLength / 1e9).toFixed(2)} GB)...`);
      await s3.send(new DeleteObjectCommand({ Bucket: RUNPOD_VOLUME_ID, Key: key }));
      console.log(`  done.`);
    } catch (err) {
      if (err.name === "NotFound" || err.$metadata?.httpStatusCode === 404) {
        console.log(`Skipping ${key} - not found (already gone).`);
      } else {
        console.error(`Failed on ${key}:`, err.message);
      }
    }
  }
  console.log("\nFree-up complete.");
}

main().catch((err) => {
  console.error("Script failed:", err);
  process.exit(1);
});
