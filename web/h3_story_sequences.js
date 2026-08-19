import { app } from "../../scripts/app.js";

const DEFAULT_HOSTS = {
  ollama: "http://localhost:11434",
  lmstudio: "http://localhost:1234",
  llamacpp: "http://localhost:8080",
};

app.registerExtension({
  name: "H3PromptStudio.StoryToSequences",
  async nodeCreated(node) {
    if (node.comfyClass !== "H3StoryToSequences") return;

    const findW = (name) => node.widgets?.find((w) => w.name === name);
    const backendW = findW("backend");
    const hostW = findW("host");
    const modelWidgets = ["vision_model", "story_model", "sequence_model"]
      .map(findW)
      .filter(Boolean);

    if (!backendW) return;

    let lastAutoHost = hostW ? hostW.value : "";

    async function refreshModels() {
      const backend = backendW.value;
      const host = hostW ? hostW.value : "";
      try {
        const url = `/h3_prompt_studio/models?backend=${encodeURIComponent(backend)}&host=${encodeURIComponent(host)}`;
        const res = await fetch(url);
        const data = await res.json();
        const models = data.models?.length ? data.models : ["(saisir un nom de modele)"];
        for (const w of modelWidgets) {
          const prev = w.value;
          w.options.values = models;
          w.value = models.includes(prev) ? prev : models[0];
        }
        node.graph?.setDirtyCanvas(true, true);
      } catch (e) {
        console.warn("H3 Prompt Studio: impossible de récupérer les modèles", e);
      }
    }

    const origBackendCallback = backendW.callback;
    backendW.callback = function (value) {
      // Ne remplace le host que s'il n'a pas été modifié manuellement par
      // l'utilisateur depuis le dernier changement de backend.
      if (hostW && (hostW.value === lastAutoHost || !hostW.value)) {
        hostW.value = DEFAULT_HOSTS[value] || hostW.value;
        lastAutoHost = hostW.value;
      }
      refreshModels();
      return origBackendCallback?.apply(this, arguments);
    };

    if (hostW) {
      const origHostCallback = hostW.callback;
      hostW.callback = function (value) {
        refreshModels();
        return origHostCallback?.apply(this, arguments);
      };
    }

    // Rafraîchissement initial (utile après un reload de page / undo-redo).
    refreshModels();
  },
});
