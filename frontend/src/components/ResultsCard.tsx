import {
  Badge,
  Box,
  Code,
  Heading,
  SimpleGrid,
  Table,
  VStack,
} from "@chakra-ui/react";
import type { ResultItem } from "../types";

type Props = {
  address: string;
  item: ResultItem;
  showJson?: boolean;
};

export default function ResultsCard({
  address,
  item,
  showJson = false,
}: Props) {
  const labels = item.labels ?? {};
  const probs = item.label_probs ?? {};
  const keys = Array.from(
    new Set([...Object.keys(labels), ...Object.keys(probs)])
  ).sort();

  const isAnomaly = item.anomaly === 1 || item.anomaly === true;
  const anomalyScore =
    typeof item.anomaly_score === "number"
      ? item.anomaly_score.toFixed(6)
      : undefined;

  return (
    <Box borderWidth="1px" borderRadius="lg" p={4} bg="white">
      <VStack align="stretch" gap={3}>
        <Heading as="h3" size="sm" display="flex" alignItems="center" gap="8px">
          <Code fontSize="0.9em">{address}</Code>
          <Badge colorScheme={isAnomaly ? "red" : "green"}>
            {isAnomaly ? "Anomaly" : "Normal"}
          </Badge>
          {anomalyScore !== undefined && (
            <Badge variant="outline">score: {anomalyScore}</Badge>
          )}
        </Heading>

        <SimpleGrid columns={{ base: 1, md: 2 }} gap={4}>
          <Box>
            <Heading as="h4" size="xs" color="gray.600" mb={2}>
              Labels
            </Heading>
            <Box display="flex" flexWrap="wrap" gap="8px">
              {keys.map((k) => {
                const v = labels[k] ?? 0;
                return (
                  <Badge
                    key={k}
                    variant={v ? "solid" : "outline"}
                    colorScheme={v ? "blue" : "gray"}
                  >
                    {k}: {v}
                  </Badge>
                );
              })}
            </Box>
          </Box>

          <Box>
            <Heading as="h4" size="xs" color="gray.600" mb={2}>
              Label probabilities
            </Heading>
            <Table.Root size="sm">
              <Table.Header>
                <Table.Row>
                  <Table.ColumnHeader>Label</Table.ColumnHeader>
                  <Table.ColumnHeader>Probability</Table.ColumnHeader>
                </Table.Row>
              </Table.Header>
              <Table.Body>
                {keys.map((k) => (
                  <Table.Row key={k}>
                    <Table.Cell>{k}</Table.Cell>
                    <Table.Cell>{Number(probs[k] ?? 0).toFixed(6)}</Table.Cell>
                  </Table.Row>
                ))}
              </Table.Body>
            </Table.Root>
          </Box>
        </SimpleGrid>

        {showJson && (
          <Box
            bg="gray.50"
            borderWidth="1px"
            borderRadius="md"
            p={3}
            overflow="auto"
          >
            <pre style={{ margin: 0, fontSize: "12px" }}>
              {JSON.stringify(item, null, 2)}
            </pre>
          </Box>
        )}
      </VStack>
    </Box>
  );
}
