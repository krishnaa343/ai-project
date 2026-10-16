import React, { useState, useEffect } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import './Navbar.css';

/**
 * Navbar — top navigation with mobile hamburger menu.
 * Uses NavLink for active-state styling.
 */
export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  // Close mobile menu on route change
  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  // Add shadow when scrolled
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`navbar${scrolled ? ' navbar--scrolled' : ''}`} aria-label="Main navigation">
      <div className="navbar__inner container">
        {/* Logo */}
        <Link to="/" className="navbar__logo" aria-label="CropScan AI — Home">
          <span className="navbar__logo-icon" aria-hidden="true">🌿</span>
          <span className="navbar__logo-text">
            CropScan <strong>AI</strong>
          </span>
        </Link>

        {/* Desktop links */}
        <ul className="navbar__links" role="list">
          <li>
            <NavLink
              to="/"
              className={({ isActive }) => `navbar__link${isActive ? ' navbar__link--active' : ''}`}
              end
            >
              Home
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/detect"
              className={({ isActive }) => `navbar__link${isActive ? ' navbar__link--active' : ''}`}
            >
              Detect Disease
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/dashboard"
              className={({ isActive }) => `navbar__link${isActive ? ' navbar__link--active' : ''}`}
            >
              Dashboard
            </NavLink>
          </li>
        </ul>

        {/* Mobile hamburger */}
        <button
          className={`navbar__hamburger${menuOpen ? ' navbar__hamburger--open' : ''}`}
          aria-expanded={menuOpen}
          aria-controls="mobile-menu"
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          onClick={() => setMenuOpen(v => !v)}
        >
          <span /><span /><span />
        </button>
      </div>

      {/* Mobile drawer */}
      <div
        id="mobile-menu"
        className={`navbar__mobile${menuOpen ? ' navbar__mobile--open' : ''}`}
        aria-hidden={!menuOpen}
      >
        <ul role="list">
          <li>
            <NavLink to="/" className={({ isActive }) => `navbar__mobile-link${isActive ? ' active' : ''}`} end>
              Home
            </NavLink>
          </li>
          <li>
            <NavLink to="/detect" className={({ isActive }) => `navbar__mobile-link${isActive ? ' active' : ''}`}>
              Detect Disease
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard" className={({ isActive }) => `navbar__mobile-link${isActive ? ' active' : ''}`}>
              Dashboard
            </NavLink>
          </li>
        </ul>
      </div>
    </nav>
  );
}
