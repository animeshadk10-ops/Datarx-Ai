import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/lib/ThemeContext";
import { SessionProvider } from "@/lib/SessionContext";

export const metadata: Metadata = {
  title: "DataRx AI — AI-Powered Data Diagnostics",
  description:
    "Upload your dataset and let AI diagnose data quality issues, recommend fixes, and export a cleaned version — powered by Gemini.",
  keywords: ["data quality", "AI", "data cleaning", "CSV", "diagnostics", "Gemini"],
  authors: [{ name: "DataRx AI" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg-primary text-text-primary antialiased">
        <ThemeProvider>
          <SessionProvider>
            {children}
          </SessionProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
