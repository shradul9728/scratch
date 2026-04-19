import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "Scratch — GPT From Scratch",
  description: "A 49.2M parameter GPT model built from scratch with PyTorch. Trained on Python code.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main style={{ paddingTop: '56px', minHeight: '100vh' }}>
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
