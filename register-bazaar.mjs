/**
 * One-time script to register this API with the CDP x402 Bazaar.
 * Run once after deploy: node register-bazaar.mjs
 *
 * What it does:
 *  1. Creates a CDP buyer wallet on Base Sepolia (testnet)
 *  2. Makes a test call to our live API (which returns 402)
 *  3. The CDP facilitator sees discoverable:true and indexes us in the Bazaar
 *
 * Requirements:
 *  npm install @coinbase/cdp-sdk x402-fetch dotenv
 */

import { CdpClient } from "@coinbase/cdp-sdk";
import { wrapFetchWithPayment } from "x402-fetch";
import dotenv from "dotenv";
dotenv.config();

const API_URL = process.env.BASE_URL || "https://memecoin-sentiment-api.onrender.com";

async function main() {
  console.log("Connecting to CDP...");
  const cdp = new CdpClient({
    apiKeyId:     process.env.CDP_API_KEY_ID,
    apiKeySecret: process.env.CDP_API_KEY_SECRET,
  });

  console.log("Getting/creating buyer wallet...");
  const account = await cdp.evm.getOrCreateAccount({ name: "bazaar-registrar" });
  console.log("Wallet address:", account.address);
  console.log("Fund this address with testnet USDC on Base Sepolia:");
  console.log("  https://faucet.circle.com  (select Base Sepolia + USDC)");
  console.log();

  const fetchWithPayment = wrapFetchWithPayment(fetch, account);

  console.log(`Pinging ${API_URL}/sentiment/DOGE ...`);
  try {
    const res = await fetchWithPayment(`${API_URL}/sentiment/DOGE`);
    const data = await res.json();
    console.log("Success! Response:", JSON.stringify(data, null, 2));
    console.log();
    console.log("CDP Bazaar registration triggered.");
    console.log("Check the Bazaar catalog in ~60s:");
    console.log("  https://api.cdp.coinbase.com/platform/v2/x402/discovery/resources");
  } catch (err) {
    if (err.message?.includes("insufficient") || err.message?.includes("balance")) {
      console.log("Wallet needs testnet USDC. Fund it at https://faucet.circle.com then re-run.");
    } else {
      console.error("Error:", err.message);
    }
  }
}

main();
