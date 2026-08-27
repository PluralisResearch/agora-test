/* Append the current page's H1 to the breadcrumb (.md-path), so it
   reads "Pluralis-8B run > Quick Start > Requirements" instead of
   stopping at the section ancestor. Material's free `navigation.path`
   only shows ancestors; the current page is normally signalled by the
   H1 below the breadcrumb, which the user found confusing because
   sibling pages in the same section produce identical breadcrumbs.

   Re-runs on every page navigation via Material's `document$` observable
   (necessary because navigation.instant skips full page reloads). */

(function () {
  function ensureCurrentInBreadcrumb() {
    const list = document.querySelector(".md-path__list");
    if (!list) return;

    // Strip any previously-injected current item (instant-nav re-entry).
    list
      .querySelectorAll(".md-path__item--current")
      .forEach((el) => el.remove());

    const h1 = document.querySelector("article.md-content__inner h1, article.md-typeset h1");
    if (!h1) return;

    // Pull the page title without anchor links / permalink markers.
    const titleNode = h1.cloneNode(true);
    titleNode
      .querySelectorAll(".headerlink, a.headerlink")
      .forEach((el) => el.remove());
    const text = titleNode.textContent.trim();
    if (!text) return;

    // Avoid duplicating the immediate section ancestor when names match
    // (e.g., the section "Quick Start" contains a page also titled "Quick Start").
    const last = list.lastElementChild;
    if (
      last &&
      last.querySelector(".md-ellipsis") &&
      last.querySelector(".md-ellipsis").textContent.trim() === text
    ) {
      last.classList.add("md-path__item--current");
      return;
    }

    const li = document.createElement("li");
    li.className = "md-path__item md-path__item--current";
    const span = document.createElement("span");
    span.className = "md-ellipsis";
    span.textContent = text;
    li.appendChild(span);
    list.appendChild(li);

    // Drop the leading "Pluralis-8B run" item from the breadcrumb so the
    // path follows the folder structure rather than starting at the site
    // index. Applies to every page (the homepage doesn't render a
    // breadcrumb at all, so there's nothing to remove there).
    const items = list.querySelectorAll(
      ".md-path__item:not(.md-path__item--current)"
    );
    for (const it of items) {
      if (/Pluralis-8B/i.test(it.textContent)) {
        it.remove();
        break;
      }
    }
  }

  // Material exposes `document$` (RxJS Observable) when navigation.instant
  // is enabled. Fall back to DOMContentLoaded for safety.
  if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
    document$.subscribe(ensureCurrentInBreadcrumb);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensureCurrentInBreadcrumb);
  } else {
    ensureCurrentInBreadcrumb();
  }
})();
