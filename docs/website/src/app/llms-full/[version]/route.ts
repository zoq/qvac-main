import { source } from '@/lib/source';
import { getLLMText } from '@/lib/get-llm-text';
import { filterPagesByVersion } from '@/lib/filter-pages-by-version';
import { VERSIONS, LATEST_VERSION } from '@/lib/versions';

export const revalidate = false;

export function generateStaticParams() {
  return VERSIONS.map((v) => ({ version: v.path }));
}

export async function GET(
  _request: Request,
  props: { params: Promise<{ version: string }> },
) {
  const { version } = await props.params;
  const pages = filterPagesByVersion(source.getPages(), version, LATEST_VERSION);
  const scan = pages.map(getLLMText);
  const scanned = await Promise.all(scan);

  return new Response(scanned.join('\n\n'));
}
