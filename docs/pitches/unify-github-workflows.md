| Approved by Technical Lead | WIP |
| :---- | :---- |
| Approved by Technical Architect | **WIP** |
| \[optional\] Review from Subash | **WIP** |

# Unify Github Workflows

## Problem

The monorepo currently contains 9 native addon packages in this CI surface area (qvac-lib-infer-llamacpp-llm, qvac-lib-infer-llamacpp-embed, qvac-lib-infer-whispercpp, qvac-lib-infer-onnx-tts, qvac-lib-infer-parakeet, qvac-lib-infer-nmtcpp, qvac-lib-infer-onnx, lib-infer-diffusion, ocr-onnx), each carrying its own GitHub Actions workflow set: prebuilds, on-pr, on-pr-close, integration tests, mobile tests, C++ tests, benchmarks, and release checks. This now adds up to dozens of addon-specific workflow files that share most of their structure.

These workflows were brought in as-is from separate repositories during the monorepo migration. While they share \~80% of their structure (same platform matrix shape, similar toolchain setup, same build/publish pipeline), each copy has drifted: different vcpkg caching strategies (S3 vs filesystem vs GH Actions cache), different ccache scopes, inconsistent runner labels (`ubuntu-22.04-arm`, `ubuntu-24.04-arm64-private`, etc.), and multiple on-PR structure variants. Toolchain setup has also drifted enough that legacy compiler pinning still appears in isolated places, even though the repo baseline has moved to LLVM/clang and modern Ubuntu runners. When a bug is found or an improvement is made in one workflow, it must be manually replicated across many others \-- a process that is slow and error-prone.

This creates real engineering cost: CI fixes routinely require touching many files, cross-cutting improvements (like upgrading a runner image or fixing a vcpkg cache issue) require coordinated multi-file PRs, and onboarding a new addon means copying and adapting an entire workflow set.

## Solution 

Consolidate the addon workflows using a reusable workflow \+ thin caller pattern, leveraging GitHub Actions' workflow\_call mechanism. The approach has three layers:

### Layer 1: Reusable Composite Actions (platform toolchain setup)

Extract the repeated platform toolchain setup (currently \~30-40 steps per prebuild workflow) into composite actions under .github/actions/:

* setup-addon-toolchain: Encapsulates LLVM-19 install (with optional GPG key import), Ubuntu compiler setup with no `gcc-13` pinning, ccache (configurable scope: arm64-only vs all-ubuntu), Windows toolchain setup (choco LLVM vs manual ccache), macOS vcpkg clone \+ Xcode selection, Android NDK configuration, and Windows cmake generator config. Parameterized by inputs like ccache-scope, windows-toolchain-mode, xcode-version.  
* setup-vcpkg-cache: Wraps the three existing caching strategies behind a single interface. Inputs: backend (filesystem | s3), s3-bucket-path, cache-key-prefix. Eliminates the current situation where some workflows use actions/cache@v5 with filesystem binary sources while others use S3 with hash tracking.  
* strip-and-verify: Consolidates the 4+ different strip-debug-symbols patterns (basic strip, strip \--strip-debug, Android llvm-strip, and the Parakeet verification steps) behind a single action with inputs platform, arch, verify (bool).

### Layer 2: Reusable Prebuild Workflow

Create a single \_reusable-prebuilds-addon.yml (the \_ prefix is a common convention for reusable-only workflows) that contains the entire prebuild pipeline:

prebuild (matrix) → merge → publish-gpr → publish-npm

Inputs (all the knobs that currently vary across the current prebuild workflows):

| Input | Type | Purpose |
| :---- | :---- | :---- |
| package-slug | string | Package directory name (e.g., qvac-lib-infer-llamacpp-llm) |
| workdir | string | Full path under packages/ |
| artifact-prefix | string | Prefix for uploaded artifacts |
| vcpkg-cache-backend | string | filesystem or s3 |
| vcpkg-s3-path | string | S3 bucket subpath (only if backend=s3) |
| extra-generate-flags | string | Additional flags for bare-make generate |
| needs-vulkan | boolean | Install Vulkan SDK |
| needs-rust-targets | boolean | Install Rust cross-compilation targets |
| needs-submodules | boolean | Init git submodules |
| extra-linux-packages | string | Additional apt packages |
| extra-macos-deps | string | Additional brew packages |
| gpr-scope-rename | string | How to rename for GPR (none, prefix-rename, scope-rename) |
| repo-name | string | For publish-npm tag creation |
| tag / publish\_target | string | Publish routing |

The platform matrix is standardized and shared. Runner labels for arm64 Linux and macOS Intel can be parameterized if they differ, but the goal is to converge on a single set. Each existing prebuild workflow (including `prebuilds-lib-infer-diffusion.yml` and `prebuilds-ocr-onnx.yml`) is then reduced to a thin caller (\~20-30 lines) that triggers on the right events and passes the correct inputs to \_reusable-prebuilds-addon.yml. This is necessary because GitHub Actions requires literal strings for `uses:` and `paths:` \-- there's no way to fully collapse to one file.

### Layer 3: Reusable On-PR Workflow

Create \_reusable-on-pr-addon.yml following the Gen 2 pattern (context job, workflow\_call with inputs, consistent merge-guard expressions). This reusable workflow contains the full PR pipeline:

release-pr-guard → context → sanity-checks \+ cpp-lint \+ cpp-tests \+ prebuild → integration-tests \+ mobile-tests → merge-guard

Key design decision: Since `uses:` in job definitions must be literal, the C++ tests, integration tests, and mobile test workflows still need to be per-package (they test different models and have different test configurations). However, their structure should be converged: a single \_reusable-cpp-tests-addon.yml parameterized by workdir, model-source (HuggingFace vs S3), and extra-cmake-flags can replace the current `cpp-tests-llm.yml`, `cpp-tests-embed.yml`, `cpp-tests-diffusion.yml`, and `cpp-test-coverage-*.yml` variants. Each package keeps a thin `on-pr-*.yml` (and matching `on-pr-close-*.yml`) as a path-filtered entry point.

### Migration Strategy

1. Start with the two llama.cpp packages (LLM and Embed) \-- they are the most similar pair and validate the reusable prebuild pattern with minimal risk.  
1. Add Whispercpp and Parakeet \-- validates private-arm runner usage and cache variation without introducing the heaviest patching complexity.  
1. Add ONNX/TTS, ONNX, Diffusion, and OCR \-- validates mixed package naming (`qvac-lib-*`, `lib-*`, and non-prefixed package dirs) and test matrix differences.  
1. Add NMTCPP last \-- it's still the most complex (bergamot patches, C++20 compatibility fixes, Windows clang-cl workarounds) and may need pre-build-hook support.  
1. Converge on-PR and on-PR-close workflows to one structure during or after prebuild unification.

### Architecture Diagram

on-pr-qvac-lib-infer-llamacpp-llm.yml   ──┐

on-pr-qvac-lib-infer-llamacpp-embed.yml ──┤

on-pr-qvac-lib-infer-whispercpp.yml     ──┤──▶ \_reusable-on-pr-addon.yml

on-pr-qvac-lib-infer-onnx-tts.yml       ──┤         │

on-pr-qvac-lib-infer-onnx.yml           ──┤         ├──▶ sanity-checks (action)

on-pr-qvac-lib-infer-parakeet.yml       ──┤         ├──▶ cpp-lint (action)

on-pr-qvac-lib-infer-nmtcpp.yml         ──┘         ├──▶ \_reusable-cpp-tests-addon.yml

on-pr-lib-infer-diffusion.yml           ──┐         │

on-pr-ocr-onnx.yml                      ──┘         │

prebuilds-qvac-lib-infer-llamacpp-llm.yml   ──┐     ├──▶ \_reusable-prebuilds-addon.yml

prebuilds-qvac-lib-infer-llamacpp-embed.yml ──┤     │       │

prebuilds-qvac-lib-infer-whispercpp.yml     ──┤     │       ├──▶ setup-addon-toolchain (action)

prebuilds-qvac-lib-infer-onnx-tts.yml       ──┼─────┘       ├──▶ setup-vcpkg-cache (action)

prebuilds-qvac-lib-infer-onnx.yml           ──┤             ├──▶ strip-and-verify (action)

prebuilds-qvac-lib-infer-parakeet.yml       ──┤             ├──▶ publish-library-to-gpr (existing)

prebuilds-qvac-lib-infer-nmtcpp.yml         ──┘             │

prebuilds-lib-infer-diffusion.yml           ──┐             │

prebuilds-ocr-onnx.yml                      ──┘             │

                                                            └──▶ publish-library-to-npm (existing)

## Risks 

| Risk | Impact | Mitigation |
| :---- | :---- | :---- |
| GitHub Actions uses: must be literal \-- can't dynamically select reusable workflows at runtime | Cannot fully collapse to a single file; thin callers are required for path-based triggers and per-package sub-workflow references | Accept this as a design constraint. Thin callers are \~20-30 lines and rarely change. The reusable workflow holds all the logic. |
| Regression in one package's build during migration \-- a subtle difference in step ordering or flag could break a specific platform | Broken prebuilds for a package, potentially blocking a release | Migrate one package pair at a time (LLM \+ Embed first). Run the old and new workflows in parallel for one PR cycle before removing the old one. Compare build artifacts byte-for-byte where feasible. |
| NMTCPP's extensive third-party patches \-- bergamot/marian-dev/intgemm C++20 compatibility patches, Windows clang-cl workarounds, and config.cmake generation are deeply embedded in its prebuild workflow | May not fit cleanly into the reusable workflow's input model; could require a "pre-build hook" escape hatch | Defer NMTCPP to last. Introduce an optional pre-build-script input on the reusable workflow that runs an arbitrary script before bare-make generate. This keeps the reusable workflow clean while accommodating outliers. |
| vcpkg cache strategy divergence \-- some packages use S3 (TTS, Parakeet), others use filesystem \+ GH Actions cache, and migrating between strategies could cause cache misses and long first builds | First build after migration could be very slow (full vcpkg rebuild), confusing developers | Keep each package's current cache strategy via the setup-vcpkg-cache action inputs. Don't force a cache strategy change \-- that's an orthogonal decision. |
| Runner label inconsistencies \-- packages use different runners for the same platform (e.g., macos-14 vs macos-15, ubuntu-22.04-arm vs ubuntu-24.04-arm64-private) | Standardizing runners could surface latent issues (different OS packages, SDK versions) | Make runner labels an input with per-package defaults. Converge runners in a separate follow-up, not during workflow unification. |
| Legacy compiler pinning sneaks back in (e.g., `gcc-13` in isolated workflows) | Divergent compiler behavior across workflows and harder reproducibility | Make compiler setup part of the shared toolchain action and explicitly ban `gcc-13` pinning in addon workflows. |
| Rabbit hole: trying to unify integration/mobile test workflows simultaneously | These workflows have significantly more per-package variation (different models, datasets, test scenarios) and could expand scope uncontrollably | Explicitly exclude integration/mobile test unification from this work. Focus on prebuilds \+ on-PR orchestration only. |

## Out of scope (optional)

* Unifying integration test workflows \-- these are inherently package-specific (different models, test scenarios, S3 assets). They remain as separate reusable workflows called from the unified on-PR workflow.  
* Unifying mobile integration test workflows \-- same reasoning; mobile tests have platform-specific tooling and device farms per package.  
* Unifying benchmark workflows \-- benchmarks are analysis-specific and have very different input shapes per package.  
* Changing vcpkg caching strategy \-- the unified workflow supports both S3 and filesystem backends; deciding which to use per package is a separate decision.  
* Runner standardization \-- converging all packages onto the same runner labels is valuable but orthogonal. The unified workflow accepts runner labels as inputs to avoid coupling.  
* Refactoring the qvac-devops external workflows \-- the external reusable workflows (release-pr-guard, public-pr merge guard) are used as-is.  
* NMTCPP's third-party dependency patches \-- the build compatibility patches for bergamot/marian-dev remain as-is, injected via the pre-build hook mechanism.

## Nice to haves (optional)

* Converge all on-PR and on-PR-close workflows to a single pattern first (context job, workflow\_call inputs, consistent merge-guard expressions) \-- this is a low-risk preparatory step that reduces the diff when reusable orchestration is introduced. High value / low effort.  
* Unified C++ test workflow (\_reusable-cpp-tests-addon.yml) \-- the current cpp-test and cpp-test-coverage workflows have very similar structure (setup, download models, build with coverage, run tests). Parameterizing by workdir and model-source could remove multiple near-duplicates. Medium value / medium effort.  
* Workflow-level CI validation \-- add a lightweight check that verifies all thin callers pass valid inputs to the reusable workflow (e.g., a JSON schema for workflow inputs validated in a CI step). Low value / low effort but prevents subtle misconfigurations.  
* Automated artifact comparison \-- a script that compares build artifacts between old and new workflow runs to detect regressions during migration. Medium value / low effort for peace of mind.

