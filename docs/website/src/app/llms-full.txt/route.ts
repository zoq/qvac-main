import { source } from '@/lib/source';
import { getLLMText } from '@/lib/get-llm-text';
import { filterPagesByVersion } from '@/lib/filter-pages-by-version';
import { LATEST_VERSION } from '@/lib/versions';

export const revalidate = false;

export async function GET() {
  const pages = filterPagesByVersion(source.getPages(), null, LATEST_VERSION);
  const scan = pages.map(getLLMText);
  const scanned = await Promise.all(scan);

  return new Response(scanned.join('\n\n'));
}
