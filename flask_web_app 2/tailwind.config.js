/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './templates/**/*.{html, js}',
  ],
  theme: {
    extend: {
      fontSize: {
        "6xl": "4rem",
        "7xl": "5rem"
      },
      fontFamily: {
        'poppins': ['Poppins', 'sans-serif']
      },
  },
  plugins: [],
}
}

