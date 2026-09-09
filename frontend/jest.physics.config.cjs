module.exports = {
  testEnvironment: "node",
  testMatch: ["<rootDir>/src/lib/__tests__/**/*.test.js"],
  transformIgnorePatterns: ["node_modules/(?!(three)/)"],
  moduleNameMapper: {
    "^three$": "<rootDir>/node_modules/three/build/three.cjs",
    "^cannon-es$": "<rootDir>/node_modules/cannon-es/dist/cannon-es.cjs.js",
  },
  transform: { "^.+\\.js$": ["babel-jest", { presets: [["@babel/preset-env", { targets: { node: "current" } }]] }] },
};