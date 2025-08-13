import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import {
  ChakraProvider,
  createSystem,
  defaultConfig,
  defineConfig,
} from "@chakra-ui/react";

import "./index.css";
import App from "./App.tsx";

const config = defineConfig({
  globalCss: {
    "#root": {
      maxW: "1280px",
      margin: "0 auto",
      padding: "2rem",
    },
  },
});

const system = createSystem(defaultConfig, config);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ChakraProvider value={system}>
      <App />
    </ChakraProvider>
  </StrictMode>
);
