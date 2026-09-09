module.exports = {
  content: ["./src/**/*.{js,jsx}", "./public/index.html"],
  theme: {
    extend: {
      colors: {
        obsidian: "#090A0F",
        panel: "#12141D",
        surface: "#1A1D2B",
        acid: "#E2F13B",
        acidhover: "#D1E228",
        muted: "#8E93A6",
        dim: "#52576B",
      },
      fontFamily: {
        display: ['"Syne"', '"Cabinet Grotesk"', "sans-serif"],
        body: ['"Plus Jakarta Sans"', '"Manrope"', "sans-serif"],
        mono: ['"JetBrains Mono"', '"Space Mono"', "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(226,241,59,.3), 0 0 40px -8px rgba(226,241,59,.4)",
        panel: "0 30px 90px -20px rgba(0,0,0,.6)",
      },
      backdropBlur: { xxl: "32px" },
    },
  },
  plugins: [],
};
