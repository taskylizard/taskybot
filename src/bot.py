import asyncio
import os

import discord
from discord.ext import commands

from .common import (
    get_messages_for_discord_thread,
)
from .inference import Inference

whitelist_dict = {}


async def keep_typing(channel):
    while True:
        await channel.trigger_typing()
        await asyncio.sleep(5)  # Trigger every 5 seconds


async def run_discord_bot():
    """Run the Discord bot that responds to mentions."""

    intents = discord.Intents.all()

    bot = commands.Bot(
        command_prefix=";",
        intents=intents,
        allowed_mentions=discord.AllowedMentions(
            everyone=False, users=False, roles=False
        ),
    )

    @bot.event
    async def on_ready():
        print(f"{bot.user} has connected to Discord!")
        print(f"Bot is in {len(bot.guilds)} guilds")

    @bot.command(name="whitelist")
    @commands.is_owner()
    async def whitelist_channel(ctx, action: str, channel: discord.TextChannel = None):
        """Whitelist management. Usage: ;whitelist add/remove/list [#channel]"""
        if channel is None:
            channel = ctx.channel

        channel_id = str(channel.id)
        guild_id = str(ctx.guild.id)
        key = f"{guild_id}:{channel_id}"

        if action.lower() == "add":
            whitelist_dict[key] = True
            await ctx.send(f"Added {channel.mention} to whitelist.")
        elif action.lower() == "remove":
            if key in whitelist_dict:
                del whitelist_dict[key]
                await ctx.send(f"Removed {channel.mention} from whitelist.")
            else:
                await ctx.send(f"{channel.mention} was not whitelisted.")
        elif action.lower() == "list":
            whitelisted = []
            for k in whitelist_dict.keys():
                try:
                    g_id, c_id = k.split(":")
                    if g_id == guild_id:
                        ch = bot.get_channel(int(c_id))
                        if ch:
                            whitelisted.append(ch.mention)
                except:
                    continue
            if whitelisted:
                await ctx.send(f"Whitelisted channels: {', '.join(whitelisted)}")
            else:
                await ctx.send("No channels are whitelisted in this server.")
        else:
            await ctx.send("Usage: `;whitelist add/remove/list [#channel]`")

    @whitelist_channel.error
    async def whitelist_error(ctx, error):
        if isinstance(error, commands.NotOwner):
            await ctx.send("Only the bot owner can use this command.")

    @bot.event
    async def on_message(message):
        # Process commands first
        await bot.process_commands(message)

        # Don't respond to ourselves
        if message.author == bot.user:
            return

        # Only respond in guild text channels, not DMs
        if not isinstance(message.channel, discord.TextChannel):
            return

        # Check if channel is whitelisted (if any channels are whitelisted for this guild)
        guild_id = str(message.guild.id)
        channel_id = str(message.channel.id)
        key = f"{guild_id}:{channel_id}"

        # Check if any channels in this guild are whitelisted
        guild_has_whitelist = any(
            k.startswith(f"{guild_id}:") for k in whitelist_dict.keys()
        )

        # If whitelist exists for this guild and this channel isn't whitelisted, don't respond
        if guild_has_whitelist and key not in whitelist_dict:
            return

        # Only respond when mentioned
        if bot.user not in message.mentions:
            return

        print(
            f"Bot mentioned in #{message.channel.name} by {message.author.display_name}"
        )

        # Use hardcoded user ID
        active_user = "352837489125228546"

        # Only include context if this is a reply to another message
        messages_in_thread = []
        typing_task = asyncio.create_task(keep_typing(message.channel))

        if message.reference and message.reference.message_id:
            try:
                referenced_message = await message.channel.fetch_message(
                    message.reference.message_id
                )
                # Only include the referenced message as context
                messages_in_thread.append(referenced_message)
            except discord.NotFound:
                pass

        # Add current message
        messages_in_thread.append(message)

        user_names = {}
        for msg in messages_in_thread:
            if hasattr(msg.author, "display_name"):
                user_names[msg.author.id] = (
                    msg.author.display_name,
                    msg.author.name,
                )

        # Convert to OpenAI chat format
        conversation = get_messages_for_discord_thread(
            messages_in_thread, bot.user.id, int(active_user), user_names
        )

        print(f"💬 Context: {len(conversation)} messages")
        print(f"Context: {conversation}")
        # Generate response
        model = Inference()

        try:
            response_text = ""
            gen = model.generate(conversation, user=active_user)
            for chunk in gen:
                response_text += chunk
            typing_task.cancel()

            if response_text:
                # Split long messages
                if len(response_text) > 2000:
                    # Discord message limit is 2000 chars
                    for i in range(0, len(response_text), 2000):
                        chunk = response_text[i : i + 2000]
                        if i == 0:
                            await message.reply(chunk)
                        else:
                            await message.channel.send(chunk)
                else:
                    await message.reply(response_text)
            else:
                await message.reply("*thinking...*")

        except Exception as e:
            print(f"Error generating response: {e}")
            await message.reply("Sorry, I encountered an error generating a response.")

    # Start the bot
    token = os.environ["DISCORD_TOKEN"]
    await bot.start(token)
