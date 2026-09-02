import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Autonomous Support AI",
  description: "RAG support chatbot with a live view of the backend pipeline.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      {/* h-screen + overflow-hidden: the two panels scroll internally, so
          the page itself never scrolls and the toolbar stays put. */}
      <body className="h-screen overflow-hidden text-slate-200 antialiased">
        {children}
      </body>
    </html>
  );
}
