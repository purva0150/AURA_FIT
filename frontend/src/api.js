import axios from "axios";

const BACKEND = process.env.REACT_APP_BACKEND_URL || "";
export const API = axios.create({ baseURL: `${BACKEND}/api` });
export const BACKEND_URL = BACKEND;

export async function fetchGarments() {
  const { data } = await API.get("/garments");
  return data.garments;
}
export async function createSession() {
  const { data } = await API.post("/sessions");
  return data; // { session, mobile_url, origin }
}
export async function getSession(token) {
  const { data } = await API.get(`/sessions/${token}`);
  return data.session;
}
export async function updateSession(token, patch) {
  const { data } = await API.post(`/sessions/${token}`, patch);
  return data.session;
}
export async function heartbeat(token, phone_name) {
  const { data } = await API.post(`/sessions/${token}/heartbeat`, { phone_name });
  return data.session;
}
export function qrUrl(token) {
  return `${BACKEND}/api/sessions/${token}/qr`;
}
