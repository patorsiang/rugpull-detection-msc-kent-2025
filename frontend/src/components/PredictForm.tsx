import { useState } from "react";
import {
  Box,
  Button,
  Field,
  Textarea,
  VStack,
  Text,
  HStack,
  Collapsible,
  Code,
} from "@chakra-ui/react";
import { parseAddresses } from "../utils/eth";
import { getCurlEndpoint } from "../utils/config";
import type { PredictApiResult } from "../types";

type Props = {
  onResult: (data: PredictApiResult) => void;
};

export default function PredictForm({ onResult }: Props) {
  const [addresses, setAddresses] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showCurl, setShowCurl] = useState(false);

  const handlePredict = async () => {
    setError(null);
    const addrs = parseAddresses(addresses);

    if (addrs.length === 0)
      return setError("Enter at least one valid 0x… address.");

    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ addresses: addrs }), // adjust if your API expects a different shape
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`);
      onResult(json);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const buildCurl = (addrs: string[]) => {
    const payload = JSON.stringify({ addresses: addrs });
    const CURL_ENDPOINT = getCurlEndpoint(); // <-- read from env/runtime
    return `curl -X POST '${CURL_ENDPOINT}' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '${payload}'`;
  };

  const handleCopyCurl = async () => {
    const addrs = parseAddresses(addresses);
    if (addrs.length === 0) {
      setError("Enter at least one valid 0x… address before copying the cURL.");
      return;
    }
    try {
      await navigator.clipboard.writeText(buildCurl(addrs));
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
      if (!showCurl) setShowCurl(true);
    } catch (e) {
      setError(`Failed to copy to clipboard.`);
      console.error(e);
    }
  };

  const parsed = parseAddresses(addresses);
  const curlPreview = parsed.length
    ? buildCurl(parsed)
    : buildCurl(["0x831467b7b6bf9c705dc87899d48b57ee55c8d5cc"]);

  return (
    <Box borderWidth="1px" borderRadius="lg" p={4} bg="white">
      <VStack align="stretch" gap={4}>
        <Collapsible.Root>
          <Field.Root id="addresses">
            <Field.Label htmlFor="addresses">Contract Addresses</Field.Label>
            <Textarea
              rows={4}
              placeholder={`0x292e89d5d5bdab3af2f5838c194c1983f0140b43\n0x1234...`}
              value={addresses}
              onChange={(e) => setAddresses(e.target.value)}
            />
            <Text fontSize="sm" color="gray.500" mt={1}>
              One per line / comma / space separated. We’ll validate and
              de‑duplicate.
            </Text>
          </Field.Root>

          {error && (
            <Text color="red.500" fontSize="sm">
              {error}
            </Text>
          )}

          <HStack>
            <Button variant="outline" onClick={handlePredict} loading={loading}>
              Predict
            </Button>
            <Button
              variant="outline"
              onClick={() =>
                setAddresses("0x292e89d5d5bdab3af2f5838c194c1983f0140b43")
              }
            >
              Load sample
            </Button>
            <Button variant="outline" onClick={handleCopyCurl}>
              {copied ? "Copied!" : "Copy cURL"}
            </Button>
            <Collapsible.Trigger onClick={() => setShowCurl((s) => !s)}>
              {showCurl ? "Hide cURL" : "Show cURL"}
            </Collapsible.Trigger>
          </HStack>

          <Collapsible.Content>
            <Box
              bg="gray.50"
              borderWidth="1px"
              borderRadius="md"
              p={3}
              overflow="auto"
              maxW="500px"
            >
              <pre style={{ margin: 0 }}>
                <Code whiteSpace="pre">{curlPreview}</Code>
              </pre>
            </Box>
          </Collapsible.Content>
        </Collapsible.Root>
      </VStack>
    </Box>
  );
}
