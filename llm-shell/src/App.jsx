
import React, { useState } from "react";
import NotesPanel from "./components/NotesPanel.jsx";
import VisualizerPanel from "./components/VisualizerPanel.jsx";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "components", label: "Components" },
  { id: "softmax", label: "Softmax" },
  { id: "notes", label: "Notes" }
];

const CONTENT = {
  overview: {
    title: "Overview",
    sections: [
      {
        id: "intro",
        title: "Introduction",
        body: `Welcome to the walkthrough shell.

This is the layout and navigation for the visualization site.`
      },
      {
        id: "preliminaries",
        title: "Preliminaries",
        body: "You can replace this with your own content for this chapter."
      }
    ]
  },
  components: {
    title: "Components",
    sections: [
      {
        id: "embedding",
        title: "Embedding",
        body: "Content for the embedding section."
      },
      {
        id: "attention",
        title: "Self-Attention",
        body: "Content for the self-attention section."
      }
    ]
  },
  softmax: {
    title: "Softmax",
    sections: [
      {
        id: "softmax-intro",
        title: "Softmax",
        body: "Content for the softmax section."
      }
    ]
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [activeSectionId, setActiveSectionId] = useState("intro");

  const isNotesTab = activeTab === "notes";
  const chapter = !isNotesTab ? CONTENT[activeTab] : null;
  const sections = chapter ? chapter.sections : [];
  const activeSection =
    chapter && sections.find((s) => s.id === activeSectionId)
      ? sections.find((s) => s.id === activeSectionId)
      : sections[0];

  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    if (tabId !== "notes") {
      const first = CONTENT[tabId].sections[0];
      setActiveSectionId(first.id);
    }
  };

  return (
    <div className="app-root">
      <Header />

      <div className="tab-bar">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={tab.id === activeTab ? "tab-button tab-active" : "tab-button"}
            onClick={() => handleTabChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* 🔹 Use a completely different layout for Notes so it's full-width */}
      {isNotesTab ? (
        <div
          className="notes-page-container"
          style={{
            padding: "1.5rem 2rem",
            width: "100%",
          }}
        >
          <NotesPanel />
        </div>
      ) : (
        <div className="main-layout">
          <aside className="sidebar">
            <h3 className="sidebar-title">Table of Contents</h3>
            <ul className="section-list">
              {sections.map((section) => (
                <li
                  key={section.id}
                  className={
                    section.id === activeSection?.id
                      ? "section-item section-active"
                      : "section-item"
                  }
                  onClick={() => setActiveSectionId(section.id)}
                >
                  {section.title}
                </li>
              ))}
            </ul>
          </aside>

          <section className="content-area">
            <ChapterView chapter={chapter} section={activeSection} />
          </section>
        </div>
      )}
    </div>
  );
}

function Header() {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button className="icon-button">&lt;</button>
        <span className="topbar-title">LLM Visualization Shell</span>
      </div>
      <button className="topbar-link">Home</button>
    </header>
  );
}

function ChapterView({ chapter, section }) {
  if (!chapter || !section) return null;
  return (
    <div className="chapter-wrapper">
      <h2 className="chapter-heading">Chapter: {chapter.title}</h2>
      <VisualizerPanel />
      <h3 className="section-heading">{section.title}</h3>
      <p className="section-body">{section.body}</p>
    </div>
  );
}
