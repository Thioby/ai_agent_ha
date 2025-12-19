# Plan: Fix Deprecated Lovelace API Access (Issue #38)

## Problem Summary

The code in `custom_components/ai_agent_ha/agent.py` uses dictionary-style access to retrieve Lovelace dashboard data, which is deprecated and will stop working in Home Assistant 2026.2.

**Current problematic code patterns:**
```python
# Line 2002-2003 in get_dashboards():
lovelace_config = self.hass.data.get(LOVELACE_DOMAIN, {})
dashboards = lovelace_config.get(CONF_DASHBOARDS, {})

# Line 2071-2085 in get_dashboard_config():
lovelace_config = self.hass.data.get(LOVELACE_DOMAIN, {})
dashboard = lovelace_config.get("default_dashboard")
dashboards = lovelace_config.get("dashboards", {})
```

## Root Cause

Home Assistant is transitioning from dictionary-style data access to property-based access for the Lovelace component data. The deprecation warning states:

> "Detected that custom integration accessed lovelace_data['dashboards'] instead of lovelace_data.dashboards. This will stop working in Home Assistant 2026.2"

## Solution

Replace dictionary-style access (`lovelace_data['dashboards']`, `lovelace_data.get(...)`) with property access (`lovelace_data.dashboards`).

## Implementation Steps

### Step 1: Fix `get_dashboards()` method (lines 1986-2039)

**Before:**
```python
lovelace_config = self.hass.data.get(LOVELACE_DOMAIN, {})
dashboards = lovelace_config.get(CONF_DASHBOARDS, {})
```

**After:**
```python
lovelace_data = self.hass.data.get(LOVELACE_DOMAIN)
if lovelace_data is None:
    return [{"error": "Lovelace not available"}]

# Use property access instead of dictionary access (required for HA 2026.2+)
dashboards = getattr(lovelace_data, "dashboards", {})
```

### Step 2: Fix `get_dashboard_config()` method (lines 2041-2101)

**Before:**
```python
lovelace_config = self.hass.data.get(LOVELACE_DOMAIN, {})
dashboard = lovelace_config.get("default_dashboard")
dashboards = lovelace_config.get("dashboards", {})
```

**After:**
```python
lovelace_data = self.hass.data.get(LOVELACE_DOMAIN)
if lovelace_data is None:
    return {"error": "Lovelace not available"}

# Use property access instead of dictionary access (required for HA 2026.2+)
if dashboard_url is None:
    dashboard = getattr(lovelace_data, "default_dashboard", None)
else:
    dashboards = getattr(lovelace_data, "dashboards", {})
    dashboard = dashboards.get(dashboard_url)
```

### Step 3: Backward Compatibility

Use `getattr()` with a default value to maintain backward compatibility with older Home Assistant versions that might still use the dictionary-style interface.

## Files to Modify

- `custom_components/ai_agent_ha/agent.py` (lines 1986-2101)

## Testing

After implementation:
1. Verify dashboards are retrieved correctly
2. Verify dashboard config retrieval works
3. Ensure no deprecation warnings are logged
4. Test with both default and custom dashboards

## References

- [GitHub Issue #861 - DW deprecated warning](https://github.com/dwainscheeren/dwains-lovelace-dashboard/issues/861)
- [Home Assistant Multiple Dashboards Documentation](https://www.home-assistant.io/dashboards/dashboards/)
