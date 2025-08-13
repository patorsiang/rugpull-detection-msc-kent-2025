import {
  Badge,
  Box,
  Code,
  Heading,
  SimpleGrid,
  Table,
  VStack,
} from "@chakra-ui/react";

type ResultItem = {
  labels: Record<string, number>;
  label_probs: Record<string, number>;
  anomaly?: number | boolean;
  anomaly_score?: number;
};

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
  const labels = item.labels || {};
  const probs = item.label_probs || {};
  const keys = Array.from(
    new Set([...Object.keys(labels), ...Object.keys(probs)])
  ).sort();
  const isAnomaly = item.anomaly === 1 || item.anomaly === true;

  return (
    <Box borderWidth="1px" borderRadius="lg" p={4} bg="white">
      <VStack align="stretch" gap={3}>
        <Heading as="h3" size="sm" display="flex" alignItems="center" gap="8px">
          <Code fontSize="0.9em">{address}</Code>
          <Badge colorPalette={isAnomaly ? "red" : "default"}>
            {isAnomaly ? "Anomaly" : "Normal"}
          </Badge>
          {"anomaly_score" in item && (
            <Badge variant="outline">
              score: {Number(item.anomaly_score).toFixed(6)}
            </Badge>
          )}
        </Heading>

        <SimpleGrid columns={[1, 2]} gap={4}>
          <Box>
            <Heading as="h4" size="xs" color="gray.600" mb={2}>
              Labels
            </Heading>
            <Box display="flex" flexWrap="wrap" gap="8px">
              {keys.map((k) => (
                <Badge
                  key={k}
                  variant={labels[k] ?? 0 ? "solid" : "outline"}
                  colorScheme={labels[k] ?? 0 ? "blue" : "gray"}
                >
                  {k}: {labels[k] ?? 0}
                </Badge>
              ))}
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
