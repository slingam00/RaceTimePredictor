import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Race Time Predictor",
  description: "Predict race times from Strava training data",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
