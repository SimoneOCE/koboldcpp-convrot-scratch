// One-off script: uploads a local koboldcpp_cublas.so directly onto Kobold
// Speed Lab's network volume, replacing the stock CUDA backend cached
// there. Reuses @aws-sdk/client-s3 already installed for the test site.
//
// Uses a multipart upload (via @aws-sdk/lib-storage) instead of a single
// PutObject - a ~200MB single PUT through RunPod's Cloudflare-fronted S3
// endpoint hit a 524 (proxy timeout) on a real run. Splitting into 10MB
// parts keeps each individual request well under that timeout and lets
// failed parts retry independently instead of restarting the whole upload.
//
// Usage:
//   node upload-convrot-so.js /path/to/koboldcpp_cublas.so
//
// Needs the same RUNPOD_S3_* vars as the test site's .env (copy
// .env.example to .env and fill in RUNPOD_S3_ENDPOINT/ACCESS_KEY/SECRET_KEY
// if you haven't already).
require("dotenv").config();
const fs = require("fs");
const path = require("path");
const { S3Client, HeadObjectCommand } = require("@aws-sdk/client-s3");
const { Upload } = require("@aws-sdk/lib-storage");

const {
  RUNPOD_S3_ENDPOINT,
  RUNPOD_S3_ACCESS_KEY,
  RUNPOD_S3_SECRET_KEY,
  RUNPOD_VOLUME_ID,
} = process.env;

const localPath = process.argv[2];
if (!localPath) {
  console.error("Usage: node upload-convrot-so.js /path/to/koboldcpp_cublas.so");
  process.exit(1);
}
if (!fs.existsSync(localPath)) {
  console.error(`File not found: ${localPath}`);
  process.exit(1);
}

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

const KEY = "koboldcpp_engine/koboldcpp_cublas.so";

async function main() {
  const stat = fs.statSync(localPath);
  console.log(`Uploading ${localPath} (${(stat.size / 1e6).toFixed(1)} MB) -> s3://${RUNPOD_VOLUME_ID}/${KEY}`);

  try {
    const before = await s3.send(new HeadObjectCommand({ Bucket: RUNPOD_VOLUME_ID, Key: KEY }));
    console.log(`Existing file on volume: ${before.ContentLength} bytes, last modified ${before.LastModified}`);
  } catch (e) {
    console.log("No existing file found at that key (or HEAD failed) - proceeding anyway.");
  }

  const body = fs.createReadStream(localPath);
  const upload = new Upload({
    client: s3,
    params: {
      Bucket: RUNPOD_VOLUME_ID,
      Key: KEY,
      Body: body,
      ContentType: "application/octet-stream",
    },
    partSize: 10 * 1024 * 1024,
    queueSize: 3,
  });

  upload.on("httpUploadProgress", (progress) => {
    if (progress.loaded && progress.total) {
      const pct = ((progress.loaded / progress.total) * 100).toFixed(1);
      process.stdout.write(`\rUploaded ${(progress.loaded / 1e6).toFixed(1)} / ${(progress.total / 1e6).toFixed(1)} MB (${pct}%)   `);
    }
  });

  await upload.done();
  process.stdout.write("\n");

  const after = await s3.send(new HeadObjectCommand({ Bucket: RUNPOD_VOLUME_ID, Key: KEY }));
  console.log(`Done. Volume now has: ${after.ContentLength} bytes, last modified ${after.LastModified}`);
  if (after.ContentLength !== stat.size) {
    console.error("WARNING: uploaded size does not match local file size - something went wrong.");
    process.exit(1);
  }
  console.log("Upload verified. Next cold worker on Kobold Speed Lab will load this file.");
}

main().catch((err) => {
  console.error("Upload failed:", err);
  process.exit(1);
});
