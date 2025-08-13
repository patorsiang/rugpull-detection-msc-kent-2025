import {
  Heading,
  Stack,
  Switch,
  Field,
  Grid,
  GridItem,
  Container,
} from "@chakra-ui/react";
import { useState } from "react";
import PredictForm from "./components/PredictForm";
import ResultsCard from "./components/ResultsCard";
import type { PredictApiResult } from "./types";

export default function App() {
  const [data, setData] = useState<PredictApiResult | null>(null);
  const [showJson, setShowJson] = useState(false);

  return (
    <Container mx="auto">
      <Grid templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }} gap="6">
        <GridItem>
          <Stack gap={6}>
            <Heading size="lg">Rugpull Detector — Predict Demo</Heading>

            <PredictForm onResult={setData} />

            <Field.Root
              display="flex"
              alignItems="center"
              w="fit-content"
              id="json-toggle"
            >
              <Field.Label htmlFor="json-toggle" mb="0">
                Show raw JSON
              </Field.Label>
              <Switch.Root
                checked={showJson}
                onCheckedChange={(e) => setShowJson(e.checked)}
              >
                <Switch.HiddenInput />
                <Switch.Control>
                  <Switch.Thumb />
                </Switch.Control>
                <Switch.Label />
              </Switch.Root>
            </Field.Root>
          </Stack>
        </GridItem>
        <GridItem>
          <Stack gap={6}>
            {data?.results &&
              Object.entries(data.results).map(([addr, item]) => (
                <ResultsCard
                  key={addr}
                  address={addr}
                  item={item}
                  showJson={showJson}
                />
              ))}
          </Stack>
        </GridItem>
      </Grid>
    </Container>
  );
}
