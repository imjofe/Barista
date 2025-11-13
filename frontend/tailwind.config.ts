import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        coffee: {
          light: "#F8EDE3",
          DEFAULT: "#C38154",
          dark: "#7F5539",
        },
      },
    },
  },
  plugins: [],
};

export default config;

