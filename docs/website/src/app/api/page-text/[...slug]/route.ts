import { source } from '@/lib/source';
import { getLLMText } from '@/lib/get-llm-text';
import { NextResponse } from 'next/server';

export const revalidate = false;

export function generateStaticParams() {
  const allPages = source.getPages().filter((page) => page.url !== '/');

  return allPages
    .filter((page) => {
      const hasChildren = allPages.some(
        (other) => other !== page && other.url.startsWith(`${page.url}/`),
      );
      return !hasChildren;
    })
    .map((page) => ({
      slug: page.url.replace(/^\//, '').split('/'),
    }));
}

export async function GET(
  _request: Request,
  props: { params: Promise<{ slug: string[] }> },
) {
  const { slug } = await props.params;
  const page = source.getPage(slug);
  if (!page) {
    return NextResponse.json({ error: 'Page not found' }, { status: 404 });
  }

  const text = await getLLMText(page);
  return new Response(text);
}
