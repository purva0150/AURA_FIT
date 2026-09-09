// Repository checks use ESLint 9; CRA retains its own ESLint 8 configuration.
const js = require("./frontend/node_modules/@eslint/js");
const globals = require("./frontend/node_modules/globals");
const react = require("./frontend/node_modules/eslint-plugin-react");

module.exports = [
  { ignores: ["**/node_modules/**", "frontend/build/**", "backend/**"] },
  js.configs.recommended,
  { linterOptions: { reportUnusedDisableDirectives: "off" } },
  {
    files: ["frontend/src/**/*.{js,jsx}"],
    languageOptions: {
      ecmaVersion: "latest", sourceType: "module",
      parserOptions: { ecmaFeatures: { jsx: true } },
      globals: { ...globals.browser, ...globals.es2021, process: "readonly" },
    },
    plugins: { react },
    rules: {
      "react/jsx-uses-react": "error",
      "react/jsx-uses-vars": "error",
      "no-unused-vars": ["error", { args: "none", caughtErrors: "none" }],
    },
  },
  {
    files: ["frontend/src/**/*Worker.js", "frontend/public/**/*.js"],
    languageOptions: { globals: { ...globals.worker, ...globals.es2021 } },
    rules: { "no-redeclare": ["error", { builtinGlobals: false }] },
  },
  {
    files: ["**/*.config.{js,cjs}", "frontend/*.js", "frontend/tests/**/*.mjs", "frontend/src/**/*.test.js"],
    languageOptions: { globals: { ...globals.node, ...globals.jest, ...globals.es2021 } },
    rules: { "no-unused-vars": ["error", { args: "none", caughtErrors: "none" }] },
  },
];