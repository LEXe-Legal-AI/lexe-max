"""Tool Health Monitor.

Monitors tool health and handles alerting for degraded tools.
"""

from datetime import datetime, timedelta
from typing import Any

import httpx
import structlog

from lexe_api.config import settings
from lexe_api.database import db
from lexe_api.models.schemas import CircuitState, ToolHealthResponse, ToolState

logger = structlog.get_logger(__name__)


class ToolHealthMonitor:
    """Monitors health of legal tools and handles alerts."""

    TOOLS = ["normattiva", "eurlex", "infolex"]

    async def get_all_health(self) -> dict[str, ToolHealthResponse]:
        """Get health status for all tools."""
        health = {}
        for tool_name in self.TOOLS:
            health[tool_name] = await self.get_tool_health(tool_name)
        return health

    async def get_tool_health(self, tool_name: str) -> ToolHealthResponse:
        """Get health status for a specific tool."""
        data = await db.get_tool_health(tool_name)

        if not data:
            return ToolHealthResponse(
                tool_name=tool_name,
                state=ToolState.HEALTHY,
                circuit_state=CircuitState.CLOSED,
            )

        return ToolHealthResponse(
            tool_name=tool_name,
            state=ToolState(data.get("state", "healthy")),
            circuit_state=CircuitState(data.get("circuit_state", "closed")),
            failure_count=data.get("failure_count", 0),
            success_count=data.get("success_count", 0),
            last_success_at=data.get("last_success_at"),
            last_failure_at=data.get("last_failure_at"),
            last_error_message=data.get("last_error_message"),
            fallback_available=True,  # Cache always available
        )

    async def report_failure(
        self,
        tool_name: str,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Report a tool failure.

        If failure count exceeds threshold, triggers alerting.
        """
        failure_count = await db.increment_tool_failure(tool_name)

        await db.update_tool_health(
            tool_name,
            last_error_message=str(error),
            last_error_type=type(error).__name__,
        )

        logger.warning(
            "Tool failure reported",
            tool=tool_name,
            error=str(error),
            failure_count=failure_count,
            context=context,
        )

        if failure_count >= settings.health_failure_threshold:
            await self._handle_threshold_breach(tool_name, error, failure_count)

    async def report_success(self, tool_name: str) -> None:
        """Report a successful tool execution."""
        await db.increment_tool_success(tool_name)

    async def _handle_threshold_breach(
        self,
        tool_name: str,
        error: Exception,
        failure_count: int,
    ) -> None:
        """Handle when a tool exceeds the failure threshold."""
        # Check if we're in alerting cooldown
        health = await db.get_tool_health(tool_name)
        last_alert = health.get("last_alert_sent_at") if health else None

        if last_alert:
            cooldown = timedelta(hours=settings.health_alert_cooldown_hours)
            if datetime.utcnow() - last_alert < cooldown:
                logger.debug(
                    "Skipping alert (cooldown)",
                    tool=tool_name,
                    last_alert=last_alert,
                )
                return

        # Update state to degraded
        await db.update_tool_health(
            tool_name,
            state="degraded",
            circuit_state="open",
            circuit_opened_at=datetime.utcnow(),
            circuit_retry_at=datetime.utcnow() + timedelta(minutes=5),
        )

        # Send alerts
        await self._notify_admins(tool_name, error, failure_count)

        # Update alert timestamp
        await db.update_tool_health(
            tool_name,
            last_alert_sent_at=datetime.utcnow(),
        )

    async def _notify_admins(
        self,
        tool_name: str,
        error: Exception,
        failure_count: int,
    ) -> None:
        """Send alert notifications to admins."""
        logger.error(
            "Tool degraded - alerting admins",
            tool=tool_name,
            error=str(error),
            failure_count=failure_count,
        )

        # Send webhook notification if configured
        if settings.alert_webhook_url:
            await self._send_webhook_alert(tool_name, error, failure_count)

        # TODO: Implement email notification via leo-core email service
        if settings.admin_email_list:
            logger.info(
                "Would send email alert",
                to=settings.admin_email_list,
                tool=tool_name,
            )

    async def _send_webhook_alert(
        self,
        tool_name: str,
        error: Exception,
        failure_count: int,
    ) -> None:
        """Send alert to webhook (Slack/Discord)."""
        payload = {
            "text": f"🚨 LEXe Tool Degraded: {tool_name}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Tool:* `{tool_name}`\n"
                        f"*Error:* `{str(error)}`\n"
                        f"*Failures:* {failure_count}\n"
                        f"*Status:* Circuit breaker OPEN",
                    },
                },
            ],
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    settings.alert_webhook_url,
                    json=payload,
                    timeout=10,
                )
                logger.info("Webhook alert sent", tool=tool_name)
        except Exception as e:
            logger.error("Failed to send webhook alert", error=str(e))

    async def reset_tool(self, tool_name: str) -> None:
        """Manually reset a tool to healthy state."""
        await db.update_tool_health(
            tool_name,
            state="healthy",
            failure_count=0,
            circuit_state="closed",
            circuit_opened_at=None,
            circuit_retry_at=None,
        )
        logger.info("Tool reset to healthy", tool=tool_name)


# Singleton instance
health_monitor = ToolHealthMonitor()
