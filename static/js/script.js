
(() => {
  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    const saved = localStorage.getItem("loanrisk-theme");
    if (saved === "dark") document.body.classList.add("dark");
    themeToggle.addEventListener("click", () => {
      document.body.classList.toggle("dark");
      localStorage.setItem("loanrisk-theme", document.body.classList.contains("dark") ? "dark" : "light");
    });
  }

  const mobileMenu = document.getElementById("mobileMenu");
  if (mobileMenu) {
    mobileMenu.addEventListener("click", () => {
      const nav = document.querySelector(".nav-links");
      if (!nav) return;
      nav.style.display = nav.style.display === "flex" ? "none" : "flex";
      nav.style.position = "absolute";
      nav.style.top = "72px";
      nav.style.left = "0";
      nav.style.right = "0";
      nav.style.background = "#0b315f";
      nav.style.padding = "12px 24px";
      nav.style.flexDirection = "column";
      nav.style.alignItems = "flex-start";
    });
  }
})();
