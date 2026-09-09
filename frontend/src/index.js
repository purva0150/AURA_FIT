import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./index.css";
import Home from "./pages/Home";
import Mirror from "./pages/Mirror";
import Remote from "./pages/Remote";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/mirror" element={<Mirror />} />
      <Route path="/remote" element={<Remote />} />
      <Route path="*" element={<Home />} />
    </Routes>
  </BrowserRouter>
);
