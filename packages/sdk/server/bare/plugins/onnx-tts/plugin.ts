import path from "bare-path";
import ttsAddonLogging from "@qvac/tts-onnx/addonLogging";
import ONNXTTS from "@qvac/tts-onnx";
import {
  definePlugin,
  defineHandler,
  ttsRequestSchema,
  ttsResponseSchema,
  ModelType,
  ttsConfigSchema,
  ADDON_TTS,
  type CreateModelParams,
  type PluginModelResult,
  type ResolveContext,
  type TtsChatterboxConfig,
  type TtsSupertonicConfig,
  type TtsChatterboxRuntimeConfig,
  type TtsSupertonicRuntimeConfig,
  type TtsRuntimeConfig,
} from "@/schemas";
import { createStreamLogger, registerAddonLogger } from "@/logging";
import {
  TtsArtifactsRequiredError,
  TtsReferenceAudioRequiredError,
} from "@/utils/errors-server";
import { textToSpeech } from "@/server/bare/plugins/onnx-tts/ops/text-to-speech";
import { attachModelExecutionMs } from "@/profiling/model-execution";
import { loadReferenceAudioAt24k } from "@/server/bare/plugins/onnx-tts/wav-helper";

async function resolveLavaSRArtifacts(
  config: TtsChatterboxConfig | TtsSupertonicConfig,
  resolve: ResolveContext["resolveModelPath"],
) {
  const paths: Record<string, string> = {};
  const promises: Promise<void>[] = [];

  const bbSrc = config.ttsEnhancerBackboneSrc;
  const shSrc = config.ttsEnhancerSpecHeadSrc;
  const dnSrc = config.ttsDenoiserSrc;

  if (bbSrc) {
    promises.push(resolve(bbSrc).then((p) => { paths["enhancerBackbonePath"] = p; }));
  }
  if (shSrc) {
    promises.push(resolve(shSrc).then((p) => { paths["enhancerSpecHeadPath"] = p; }));
  }
  if (dnSrc) {
    promises.push(resolve(dnSrc).then((p) => { paths["denoiserPath"] = p; }));
  }

  await Promise.all(promises);
  return paths;
}

async function resolveChatterboxConfig(
  config: TtsChatterboxConfig,
  ctx: ResolveContext,
) {
  const {
    ttsTokenizerSrc,
    ttsSpeechEncoderSrc,
    ttsEmbedTokensSrc,
    ttsConditionalDecoderSrc,
    ttsLanguageModelSrc,
    referenceAudioSrc,
    language,
    enhance,
    denoise,
    outputSampleRate,
  } = config;

  if (
    !ttsTokenizerSrc ||
    !ttsSpeechEncoderSrc ||
    !ttsEmbedTokensSrc ||
    !ttsConditionalDecoderSrc ||
    !ttsLanguageModelSrc
  ) {
    throw new TtsArtifactsRequiredError();
  }
  if (!referenceAudioSrc) {
    throw new TtsReferenceAudioRequiredError();
  }

  const resolve = ctx.resolveModelPath;
  const [
    tokenizerPath,
    speechEncoderPath,
    embedTokensPath,
    conditionalDecoderPath,
    languageModelPath,
    referenceAudioPath,
  ] = await Promise.all([
    resolve(ttsTokenizerSrc),
    resolve(ttsSpeechEncoderSrc),
    resolve(ttsEmbedTokensSrc),
    resolve(ttsConditionalDecoderSrc),
    resolve(ttsLanguageModelSrc),
    resolve(referenceAudioSrc),
  ]);

  const lavaSRArtifacts = await resolveLavaSRArtifacts(config, resolve);

  return {
    config: {
      ttsEngine: "chatterbox",
      language,
      ...(enhance !== undefined && { enhance }),
      ...(denoise !== undefined && { denoise }),
      ...(outputSampleRate !== undefined && { outputSampleRate }),
    } as TtsChatterboxRuntimeConfig,
    artifacts: {
      tokenizerPath,
      speechEncoderPath,
      embedTokensPath,
      conditionalDecoderPath,
      languageModelPath,
      referenceAudioPath,
      ...lavaSRArtifacts,
    },
  };
}

async function resolveSupertonicConfig(
  config: TtsSupertonicConfig,
  ctx: ResolveContext,
) {
  const {
    ttsTokenizerSrc,
    ttsTextEncoderSrc,
    ttsLatentDenoiserSrc,
    ttsVoiceDecoderSrc,
    ttsVoiceSrc,
    ttsSpeed,
    ttsNumInferenceSteps,
    language,
    enhance,
    denoise,
    outputSampleRate,
  } = config;

  if (
    !ttsTokenizerSrc ||
    !ttsTextEncoderSrc ||
    !ttsLatentDenoiserSrc ||
    !ttsVoiceDecoderSrc ||
    !ttsVoiceSrc
  ) {
    throw new TtsArtifactsRequiredError();
  }

  const resolve = ctx.resolveModelPath;
  const [
    tokenizerPath,
    textEncoderPath,
    latentDenoiserPath,
    voiceDecoderPath,
    voicePath,
  ] = await Promise.all([
    resolve(ttsTokenizerSrc),
    resolve(ttsTextEncoderSrc),
    resolve(ttsLatentDenoiserSrc),
    resolve(ttsVoiceDecoderSrc),
    resolve(ttsVoiceSrc),
  ]);

  const lavaSRArtifacts = await resolveLavaSRArtifacts(config, resolve);

  return {
    config: {
      ttsEngine: "supertonic",
      language,
      ttsSpeed,
      ttsNumInferenceSteps,
      ...(enhance !== undefined && { enhance }),
      ...(denoise !== undefined && { denoise }),
      ...(outputSampleRate !== undefined && { outputSampleRate }),
    } as TtsSupertonicRuntimeConfig,
    artifacts: {
      tokenizerPath,
      textEncoderPath,
      latentDenoiserPath,
      voiceDecoderPath,
      voicePath,
      ...lavaSRArtifacts,
    },
  };
}

function createChatterboxModel(
  modelId: string,
  config: TtsChatterboxRuntimeConfig,
  artifacts: Record<string, string | undefined>,
): PluginModelResult {
  const tokenizerPath = artifacts["tokenizerPath"];
  const speechEncoderPath = artifacts["speechEncoderPath"];
  const embedTokensPath = artifacts["embedTokensPath"];
  const conditionalDecoderPath = artifacts["conditionalDecoderPath"];
  const languageModelPath = artifacts["languageModelPath"];
  const referenceAudioPath = artifacts["referenceAudioPath"];

  if (
    !tokenizerPath ||
    !speechEncoderPath ||
    !embedTokensPath ||
    !conditionalDecoderPath ||
    !languageModelPath
  ) {
    throw new TtsArtifactsRequiredError();
  }
  if (!referenceAudioPath) {
    throw new TtsReferenceAudioRequiredError();
  }

  const logger = createStreamLogger(modelId, ModelType.onnxTts);
  registerAddonLogger(modelId, ModelType.onnxTts, logger);
  const referenceAudio = loadReferenceAudioAt24k(referenceAudioPath);
  const args: Record<string, unknown> = {
    tokenizerPath,
    speechEncoderPath,
    embedTokensPath,
    conditionalDecoderPath,
    languageModelPath,
    referenceAudio,
    logger,
    opts: { stats: true },
  };

  if (config.enhance !== undefined) args["enhance"] = config.enhance;
  if (config.denoise !== undefined) args["denoise"] = config.denoise;
  if (config.outputSampleRate !== undefined) args["outputSampleRate"] = config.outputSampleRate;
  if (artifacts["enhancerBackbonePath"]) args["enhancerBackbonePath"] = artifacts["enhancerBackbonePath"];
  if (artifacts["enhancerSpecHeadPath"]) args["enhancerSpecHeadPath"] = artifacts["enhancerSpecHeadPath"];
  if (artifacts["denoiserPath"]) args["denoiserPath"] = artifacts["denoiserPath"];

  const modelConfig = { language: config.language ?? "en", useGPU: false };
  const model = new ONNXTTS(args as never, modelConfig);
  return { model, loader: undefined };
}

function createSupertonicModel(
  modelId: string,
  config: TtsSupertonicRuntimeConfig,
  artifacts: Record<string, string | undefined>,
): PluginModelResult {
  const tokenizerPath = artifacts["tokenizerPath"];
  const textEncoderPath = artifacts["textEncoderPath"];
  const latentDenoiserPath = artifacts["latentDenoiserPath"];
  const voiceDecoderPath = artifacts["voiceDecoderPath"];
  const voicePath = artifacts["voicePath"];

  if (
    !tokenizerPath ||
    !textEncoderPath ||
    !latentDenoiserPath ||
    !voiceDecoderPath ||
    !voicePath
  ) {
    throw new TtsArtifactsRequiredError();
  }

  const logger = createStreamLogger(modelId, ModelType.onnxTts);
  registerAddonLogger(modelId, ModelType.onnxTts, logger);
  const voicesDir = path.dirname(voicePath);
  const voiceName = path.basename(voicePath).replace(/\.bin$/i, "") || "voice";
  const args: Record<string, unknown> = {
    tokenizerPath,
    textEncoderPath,
    latentDenoiserPath,
    voiceDecoderPath,
    voicesDir,
    voiceName,
    speed: config.ttsSpeed ?? 1,
    numInferenceSteps: config.ttsNumInferenceSteps ?? 5,
    logger,
    opts: { stats: true },
  };

  if (config.enhance !== undefined) args["enhance"] = config.enhance;
  if (config.denoise !== undefined) args["denoise"] = config.denoise;
  if (config.outputSampleRate !== undefined) args["outputSampleRate"] = config.outputSampleRate;
  if (artifacts["enhancerBackbonePath"]) args["enhancerBackbonePath"] = artifacts["enhancerBackbonePath"];
  if (artifacts["enhancerSpecHeadPath"]) args["enhancerSpecHeadPath"] = artifacts["enhancerSpecHeadPath"];
  if (artifacts["denoiserPath"]) args["denoiserPath"] = artifacts["denoiserPath"];

  const modelConfig = { language: config.language ?? "en" };
  const model = new ONNXTTS(args as never, modelConfig);
  return { model, loader: undefined };
}

export const ttsPlugin = definePlugin({
  modelType: ModelType.onnxTts,
  displayName: "TTS (ONNX)",
  addonPackage: ADDON_TTS,
  loadConfigSchema: ttsConfigSchema,
  skipPrimaryModelPathValidation: true,

  async resolveConfig(
    cfg: Record<string, unknown>,
    ctx: ResolveContext,
  ) {
    const { ttsEngine } = cfg as { ttsEngine?: string };

    if (ttsEngine === "supertonic") {
      return resolveSupertonicConfig(cfg as TtsSupertonicConfig, ctx);
    }
    return resolveChatterboxConfig(cfg as TtsChatterboxConfig, ctx);
  },

  createModel(params: CreateModelParams): PluginModelResult {
    const config = (params.modelConfig ?? {}) as TtsRuntimeConfig;
    const artifacts = params.artifacts ?? {};

    if (config.ttsEngine === "supertonic") {
      return createSupertonicModel(params.modelId, config, artifacts);
    }

    return createChatterboxModel(params.modelId, config, artifacts);
  },

  handlers: {
    textToSpeech: defineHandler({
      requestSchema: ttsRequestSchema,
      responseSchema: ttsResponseSchema,
      streaming: true,

      handler: async function* (request) {
        const stream = textToSpeech(request);
        try {
          let result = await stream.next();

          while (!result.done) {
            yield {
              type: "textToSpeech" as const,
              buffer: result.value.buffer,
              done: false,
              ...(result.value.sampleRate !== undefined && { sampleRate: result.value.sampleRate }),
            };
            result = await stream.next();
          }

          const { modelExecutionMs, stats } = result.value;
          yield attachModelExecutionMs({
            type: "textToSpeech" as const,
            buffer: [],
            done: true,
            ...(stats && { stats }),
          }, modelExecutionMs);
        } finally {
          await stream.return?.(undefined as never);
        }
      },
    }),
  },

  logging: {
    module: ttsAddonLogging,
    namespace: ModelType.onnxTts,
  },
});
