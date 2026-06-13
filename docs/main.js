(function () {
  "use strict";

  function loadSrc(video) {
    if (video.dataset.src) {
      video.src = video.dataset.src;
      video.removeAttribute("data-src");
    }
  }

  // Click toggles native controls + play/pause (controls hidden until clicked).
  document.querySelectorAll("video.click-to-control").forEach((video) => {
    video.addEventListener("click", () => {
      loadSrc(video);
      if (!video.controls) video.controls = true;
      if (video.paused) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  });

  // Autoplay (muted) while on screen, pause when scrolled away.
  const autoVideos = document.querySelectorAll("video.autoplay-inview");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const v = entry.target;
          if (entry.isIntersecting) {
            loadSrc(v);
            v.play().catch(() => {});
          } else if (!v.controls) {
            // Don't fight the user: only auto-pause videos they haven't taken over.
            v.pause();
          }
        });
      },
      { threshold: 0.2 }
    );
    autoVideos.forEach((v) => io.observe(v));
  } else {
    autoVideos.forEach((v) => {
      loadSrc(v);
      v.play().catch(() => {});
    });
  }

  // Tasks carousel
  const carousel = document.querySelector(".carousel");
  if (!carousel) return;

  const track = carousel.querySelector(".carousel-track");
  const slides = carousel.querySelectorAll(".carousel-slide");
  const prevBtn = carousel.querySelector(".carousel-btn.prev");
  const nextBtn = carousel.querySelector(".carousel-btn.next");
  const dotsContainer = carousel.querySelector(".carousel-dots");

  let current = 0;

  slides.forEach((_, i) => {
    const dot = document.createElement("button");
    dot.className = "carousel-dot" + (i === 0 ? " active" : "");
    dot.setAttribute("aria-label", "Go to slide " + (i + 1));
    dot.addEventListener("click", () => goTo(i));
    dotsContainer.appendChild(dot);
  });

  const dots = dotsContainer.querySelectorAll(".carousel-dot");

  function playSlide(slide) {
    slide.querySelectorAll("video").forEach((v) => {
      loadSrc(v);
      v.play().catch(() => {});
    });
  }

  function pauseSlide(slide) {
    slide.querySelectorAll("video").forEach((v) => {
      if (!v.controls) v.pause();
    });
  }

  function goTo(index) {
    current = ((index % slides.length) + slides.length) % slides.length;
    track.style.transform = "translateX(-" + current * 100 + "%)";
    dots.forEach((d, i) => d.classList.toggle("active", i === current));
    slides.forEach((s, i) => (i === current ? playSlide(s) : pauseSlide(s)));
  }

  prevBtn.addEventListener("click", () => goTo(current - 1));
  nextBtn.addEventListener("click", () => goTo(current + 1));

  document.addEventListener("keydown", (e) => {
    if (
      !carousel.matches(":hover") &&
      document.activeElement !== prevBtn &&
      document.activeElement !== nextBtn
    ) {
      return;
    }
    if (e.key === "ArrowLeft") goTo(current - 1);
    if (e.key === "ArrowRight") goTo(current + 1);
  });

  // Kick off the first slide.
  playSlide(slides[0]);
})();
