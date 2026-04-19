import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "Scratch — GPT From Scratch",
  description: "A complete GPT Transformer model built from scratch using PyTorch. Explore the architecture, train on code, and interact with the model.",
  keywords: ["GPT", "Transformer", "PyTorch", "AI", "Machine Learning", "LLM", "From Scratch"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <Navbar />
        <main className="pt-16 min-h-screen">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
