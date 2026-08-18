import { copyFile, mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PINNED_PACKAGES = Object.freeze({
  preact: "10.22.0",
  htm: "3.1.1",
});

const VENDOR_FILES = Object.freeze([
  {
    packageName: "preact",
    source: "dist/preact.module.js",
    destination: "preact.module.js",
  },
  {
    packageName: "preact",
    source: "hooks/dist/hooks.module.js",
    destination: "preact-hooks.module.js",
  },
  {
    packageName: "htm",
    source: "dist/htm.module.js",
    destination: "htm.module.js",
  },
]);

function repositoryRoot() {
  const rootIndex = process.argv.indexOf("--root");
  if (rootIndex >= 0) {
    const value = process.argv[rootIndex + 1];
    if (!value) throw new Error("--root requires a directory path");
    return path.resolve(value);
  }
  return path.resolve(fileURLToPath(new URL("../../", import.meta.url)));
}

async function installedVersion(root, packageName) {
  const packageJsonPath = path.join(
    root,
    "node_modules",
    packageName,
    "package.json",
  );
  const packageJson = JSON.parse(await readFile(packageJsonPath, "utf8"));
  return String(packageJson.version || "");
}

async function main() {
  const root = repositoryRoot();
  for (const [packageName, expectedVersion] of Object.entries(PINNED_PACKAGES)) {
    const actualVersion = await installedVersion(root, packageName);
    if (actualVersion !== expectedVersion) {
      throw new Error(
        `${packageName} must be exactly ${expectedVersion}; installed ${actualVersion || "unknown"}`,
      );
    }
  }

  const destinationRoot = path.join(
    root,
    "src",
    "avaturn_live_streamer",
    "vendor",
  );
  await mkdir(destinationRoot, { recursive: true });
  for (const file of VENDOR_FILES) {
    await copyFile(
      path.join(root, "node_modules", file.packageName, file.source),
      path.join(destinationRoot, file.destination),
    );
  }
  process.stdout.write(`Vendored ${VENDOR_FILES.length} frontend modules to ${destinationRoot}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
