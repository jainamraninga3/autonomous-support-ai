"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { normalizeMarkdown } from "@/lib/markdown";

/**
 * Renders an answer's markdown with explicit styles for every element the
 * model actually uses.
 *
 * Tailwind's preflight strips default element styling, so without these a
 * table renders as run-together text and a list loses its bullets — which
 * looks worse than the raw markdown did. `remark-gfm` is what turns
 * `|`-delimited rows into real tables and `---` into a rule.
 *
 * Raw HTML is intentionally NOT enabled. Answer text comes from an LLM
 * reading retrieved documents, so it is untrusted input; allowing HTML
 * here would let document content inject markup into the page.
 */
export function Markdown({ content }: { content: string }) {
  return (
    <div className="space-y-3 text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="whitespace-pre-wrap break-words">{children}</p>,

          strong: ({ children }) => (
            <strong className="font-semibold text-white">{children}</strong>
          ),
          em: ({ children }) => <em className="italic text-slate-300">{children}</em>,

          ul: ({ children }) => (
            <ul className="ml-1 list-disc space-y-1 pl-4 marker:text-slate-500">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="ml-1 list-decimal space-y-1 pl-4 marker:text-slate-500">{children}</ol>
          ),
          li: ({ children }) => <li className="break-words pl-0.5">{children}</li>,

          h1: ({ children }) => (
            <h3 className="mt-1 text-[15px] font-semibold text-white">{children}</h3>
          ),
          h2: ({ children }) => (
            <h3 className="mt-1 text-[15px] font-semibold text-white">{children}</h3>
          ),
          h3: ({ children }) => (
            <h4 className="mt-1 text-sm font-semibold text-slate-100">{children}</h4>
          ),
          h4: ({ children }) => (
            <h5 className="mt-1 text-sm font-semibold text-slate-200">{children}</h5>
          ),

          // Wide tables scroll inside their own container rather than
          // stretching the chat bubble past the panel edge.
          table: ({ children }) => (
            <div className="-mx-1 overflow-x-auto py-1">
              <table className="w-full border-collapse overflow-hidden rounded-lg border border-white/10 text-left text-[13px]">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-white/[0.06]">{children}</thead>,
          tbody: ({ children }) => (
            <tbody className="divide-y divide-white/[0.06]">{children}</tbody>
          ),
          tr: ({ children }) => <tr className="align-top">{children}</tr>,
          th: ({ children }) => (
            <th className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-300">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-slate-300">{children}</td>
          ),

          code: ({ children, className }) => {
            const isBlock = /language-/.test(className ?? "");
            if (isBlock) {
              return (
                <code className="block overflow-x-auto rounded-lg bg-slate-950/70 p-3 font-mono text-xs text-sky-200">
                  {children}
                </code>
              );
            }
            return (
              <code className="rounded bg-white/10 px-1 py-0.5 font-mono text-[12px] text-sky-200">
                {children}
              </code>
            );
          },
          pre: ({ children }) => <pre className="overflow-x-auto">{children}</pre>,

          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-sky-500/40 pl-3 text-slate-400">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="border-white/10" />,
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sky-300 underline decoration-sky-500/40 hover:decoration-sky-400"
            >
              {children}
            </a>
          ),
        }}
      >
        {normalizeMarkdown(content)}
      </ReactMarkdown>
    </div>
  );
}
