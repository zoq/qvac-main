import type { Metadata } from 'next';

/**
 * Whether the static export should declare `robots: index` in HTML.
 * Defaults to **noindex** so crawlers that never run JavaScript still omit
 * preview, PR, and local builds. Pair with the root layout hostname script so
 * a production artifact served on a non-production host stays non-indexable.
 *
 * Enable indexing only for the deploy that serves `https://docs.qvac.tether.io`:
 * - Vercel production: `VERCEL_ENV=production` is set automatically.
 * - Other CI: set `DOCS_ALLOW_INDEXING=1` on that build only.
 *
 * Optional: `DOCS_FORCE_NOINDEX=1` forces noindex even on Vercel production.
 */
export function allowDocsIndexingAtBuildTime() {
  if (process.env.DOCS_FORCE_NOINDEX === '1' || process.env.DOCS_FORCE_NOINDEX === 'true') {
    return false;
  }
  if (process.env.DOCS_ALLOW_INDEXING === '1' || process.env.DOCS_ALLOW_INDEXING === 'true') {
    return true;
  }
  return process.env.VERCEL_ENV === 'production';
}

export function docsRootMetadataRobots(): Metadata['robots'] {
  return allowDocsIndexingAtBuildTime()
    ? { index: true, follow: true }
    : { index: false, follow: false };
}
