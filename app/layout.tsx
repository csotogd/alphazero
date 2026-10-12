import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {
  title: 'Pentomino — a little space for strategy',
  description: 'Twelve pieces. Endless possibilities. A thoughtful two-player pentomino game. Play side by side or invite a friend online.',
  icons: {icon:'/favicon.svg'},
};
export default function RootLayout({children}:{children:React.ReactNode}) {
  return <html lang="en"><body>{children}</body></html>;
}
