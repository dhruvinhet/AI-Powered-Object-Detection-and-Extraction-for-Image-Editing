document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const navbar = document.querySelector('.navbar');
    const menuToggle = document.getElementById('menu-toggle');
    const currentTheme = localStorage.getItem('theme') || 'light';
    
    if (currentTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }

    // Error handling for failed resource loading
    window.addEventListener('error', (event) => {
        if (event.target.tagName === 'SCRIPT' || event.target.tagName === 'LINK') {
            console.error(`Failed to load resource: ${event.target.src || event.target.href}`);
        }
    });

    themeToggle.addEventListener('click', () => {
        const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    });

    // Particle Animation (Smooth Moving)
    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particlesArray = [];
    const numberOfParticles = 100;

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 3 + 1;
            this.speedX = Math.random() * 1 - 0.5; // Slower speed
            this.speedY = Math.random() * 1 - 0.5; // Slower speed
        }
        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            if (this.x < 0) this.x = canvas.width;
            if (this.x > canvas.width) this.x = 0;
            if (this.y < 0) this.y = canvas.height;
            if (this.y > canvas.height) this.y = 0;
        }
        draw() {
            ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--primary-color').trim();
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function init() {
        for (let i = 0; i < numberOfParticles; i++) {
            particlesArray.push(new Particle());
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < particlesArray.length; i++) {
            particlesArray[i].update();
            particlesArray[i].draw();
        }
        requestAnimationFrame(animate);
    }

    init();
    animate();

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }); 
});

// Replace the existing updateFlowchartLine function
function animateFlowchartSteps() {
    const totalDuration = 5000; // 5 seconds for the full cycle
    const steps = document.querySelectorAll('.flowchart-step');
    
    const progress = (Date.now() % totalDuration) / totalDuration;
    
    // Calculate timing for each step
    steps.forEach((step, index) => {
        const stepTiming = (index + 1) / steps.length;
        if (progress >= stepTiming - 0.2) { // Show step slightly before its time
            step.classList.add('visible');
        } else {
            step.classList.remove('visible');
        }
    });

    requestAnimationFrame(animateFlowchartSteps);
}

animateFlowchartSteps();

// Replace the existing typing effect code at the end of script.js
document.addEventListener('DOMContentLoaded', () => {
    // Existing code (theme toggle, particle animation, etc.) remains unchanged until here...

    const typingContainer = document.getElementById('typing-container');
    const creators = [
        "Dhruvinkumar Patel - 21",
        "Sanchit Sovale - 27",
        "Harshvardhan Salunke - 13",
        "Devansh Singh - 04"
    ];
    let creatorIndex = 0;
    let charIndex = 0;
    let currentSpan = null;

    function type() {
        // If all creators are typed, stop
        if (creatorIndex >= creators.length) return;

        // Create a new span for the current creator if starting a new name
        if (charIndex === 0) {
            currentSpan = document.createElement('span');
            currentSpan.className = 'typing-text';
            typingContainer.appendChild(currentSpan);
        }

        const currentCreator = creators[creatorIndex];
        const displayText = currentCreator.substring(0, charIndex++);

        // Update the current span with the typed text
        currentSpan.textContent = displayText;

        // If the current name is fully typed, move to the next creator
        if (charIndex > currentCreator.length) {
            currentSpan.classList.add('completed'); // Remove cursor
            charIndex = 0; // Reset for next name
            creatorIndex++; // Move to next creator
            setTimeout(type, 1000); // Pause before typing next name
        } else {
            setTimeout(type, 100); // Typing speed
        }
    }

    // Start the typing effect
    type();
});