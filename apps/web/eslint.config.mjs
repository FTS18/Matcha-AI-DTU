import typescriptParser from "@typescript-eslint/parser";

const eslintConfig = [
  {
    ignores: [
      ".next/**",
      "node_modules/**",
      ".turbo/**",
      "dist/**",
      "build/**",
    ],
  },
  {
    files: ["**/*.{js,mjs,jsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        window: true,
        document: true,
        navigator: true,
        fetch: true,
        process: true,
        Buffer: true,
        __dirname: true,
        __filename: true,
        global: true,
        require: true,
        module: true,
        exports: true,
        self: true,
      },
    },
    rules: {
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "no-undef": "warn",
    },
  },
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      parser: typescriptParser,
      parserOptions: {
        ecmaVersion: 2022,
        sourceType: "module",
        ecmaFeatures: {
          jsx: true,
        },
        project: "./tsconfig.json",
      },
      globals: {
        window: true,
        document: true,
        navigator: true,
        fetch: true,
        process: true,
        Buffer: true,
        __dirname: true,
        __filename: true,
        global: true,
        require: true,
        module: true,
        exports: true,
        self: true,
      },
    },
    rules: {
      "no-unused-vars": "off",
      "no-undef": "off",
    },
  },
];

export default eslintConfig;
