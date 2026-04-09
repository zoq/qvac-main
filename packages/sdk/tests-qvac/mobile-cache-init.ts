/**
 * Mobile init hook for pre-cached models on Device Farm.
 *
 * When cache-models is enabled in CI, models are pushed to
 * /data/local/tmp/qvac-models/ via adb before the app starts.
 * This hook detects that directory and writes a qvac.config.json
 * pointing cacheDirectory there, so the SDK skips downloading.
 *
 * When cache-models is disabled (or running locally), the directory
 * won't exist and this hook does nothing — models download normally.
 *
 * Referenced from qvac-test.config.js as mobileInit.
 */
import { File, Directory, Paths } from "expo-file-system";

const PRECACHED_MODELS_DIR = "file:///data/local/tmp/qvac-models";

try {
  const modelsDir = new Directory(PRECACHED_MODELS_DIR);

  if (modelsDir.exists) {
    const configFile = new File(Paths.document, "qvac.config.json");
    configFile.create({ overwrite: true });
    configFile.write(
      JSON.stringify({ cacheDirectory: "/data/local/tmp/qvac-models" }),
    );
    console.log("Pre-cached models found at /data/local/tmp/qvac-models");
  }
} catch {
  // No pre-cached models directory — SDK will download normally
}

export const __sdkPreload = true;
