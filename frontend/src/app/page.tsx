import Image from "next/image";
import { ThemeToggle } from "../components/ThemeToggle";

export default function Home() {
  return (
    <>
      <nav>
        <ThemeToggle />
      </nav>
      <h1>Hello Dashboard</h1>

    </>
  );
}
