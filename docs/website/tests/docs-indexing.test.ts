import { describe, it, expect, afterEach, vi } from 'vitest';
import { allowDocsIndexingAtBuildTime, docsRootMetadataRobots } from '@/lib/docs-indexing';

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('allowDocsIndexingAtBuildTime', () => {
  it('is false when no relevant env is set', () => {
    vi.stubEnv('VERCEL_ENV', '');
    vi.stubEnv('DOCS_ALLOW_INDEXING', '');
    vi.stubEnv('DOCS_FORCE_NOINDEX', '');
    expect(allowDocsIndexingAtBuildTime()).toBe(false);
  });

  it('is true on Vercel production', () => {
    vi.stubEnv('VERCEL_ENV', 'production');
    expect(allowDocsIndexingAtBuildTime()).toBe(true);
  });

  it('is false on Vercel preview', () => {
    vi.stubEnv('VERCEL_ENV', 'preview');
    expect(allowDocsIndexingAtBuildTime()).toBe(false);
  });

  it('is true when DOCS_ALLOW_INDEXING=1', () => {
    vi.stubEnv('DOCS_ALLOW_INDEXING', '1');
    expect(allowDocsIndexingAtBuildTime()).toBe(true);
  });

  it('is false when DOCS_FORCE_NOINDEX=1 even if production', () => {
    vi.stubEnv('VERCEL_ENV', 'production');
    vi.stubEnv('DOCS_FORCE_NOINDEX', '1');
    expect(allowDocsIndexingAtBuildTime()).toBe(false);
  });
});

describe('docsRootMetadataRobots', () => {
  it('matches allow flag', () => {
    vi.stubEnv('VERCEL_ENV', 'production');
    expect(docsRootMetadataRobots()).toEqual({ index: true, follow: true });
  });

  it('emits noindex when not allowed', () => {
    vi.stubEnv('VERCEL_ENV', 'preview');
    expect(docsRootMetadataRobots()).toEqual({ index: false, follow: false });
  });
});
