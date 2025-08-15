import { useEffect, useState } from "react";
import {
  Box,
  Container,
  Grid,
  GridItem,
  Heading,
  Stack,
  HStack,
  Switch,
  Field,
} from "@chakra-ui/react";
import PredictForm from "./components/PredictForm";
import ResultsCard from "./components/ResultsCard";
import type { PredictApiResult } from "./types";

export default function App() {
  const [data, setData] = useState<PredictApiResult | null>(null);
  const [showJson, setShowJson] = useState(false);

  // Health ping (kept lightweight and eslint-friendly)
  useEffect(() => {
    // Fire-and-forget, ignore result
    void fetch("/api/system/health").catch(() => undefined);
  }, []);

  return (
    <Container mx="auto" py={6}>
      <Grid templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }} gap={6}>
        <GridItem>
          <Stack gap={6}>
            <Heading size="lg">Rugpull Detector — Predict Demo</Heading>

            <PredictForm onResult={setData} />

            <Field.Root
              display="inline-flex"
              alignItems="center"
              w="fit-content"
            >
              <HStack gap={3}>
                <Switch.Root
                  size="lg"
                  checked={showJson}
                  onCheckedChange={(e) => setShowJson(e.checked)}
                >
                  <Switch.HiddenInput />
                  <Switch.Control />
                  <Switch.Label>Show raw JSON</Switch.Label>
                </Switch.Root>
              </HStack>
            </Field.Root>
          </Stack>
        </GridItem>

        <GridItem>
          <Stack gap={6}>
            {data?.results && showJson ? (
              <Box
                bg="gray.50"
                borderWidth="1px"
                borderRadius="md"
                p={3}
                overflow="auto"
              >
                <pre style={{ margin: 0, fontSize: "12px" }}>
                  {JSON.stringify(data, null, 2)}
                </pre>
              </Box>
            ) : (
              data?.results &&
              Object.entries(data.results).map(([addr, item]) => (
                <ResultsCard key={addr} address={addr} item={item} />
              ))
            )}
          </Stack>
        </GridItem>
      </Grid>
    </Container>
  );
}
