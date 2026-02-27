"""
transcript_builder.py
Generates turn-by-turn dialogue for each call scenario.
Each transcript is a list of dicts: {"speaker": "Agent"|"Customer", "text": str}

Credit injection rules:
  - credit_info dict is passed in from generate_calls.py after build_credit() runs
  - Each body function that can issue a credit receives credit_info and injects
    the exact dollar amount and type into the agent's dialogue
  - This ensures transcript text matches the structured metadata columns exactly
"""
import numpy as np
from typing import List, Dict


Turn = Dict[str, str]


# ---------------------------------------------------------------------------
# Shared openers / closers
# ---------------------------------------------------------------------------

def _opener(agent_name: str, customer_name: str,
            account_id: str, rng: np.random.Generator) -> List[Turn]:
    greetings = [
        f"Thank you for calling NovaWireless, this is {agent_name}. How can I help you today?",
        f"NovaWireless, you've reached {agent_name}. What can I do for you?",
        f"Thank you for calling NovaWireless. My name is {agent_name}. How can I assist you?",
    ]
    return [
        {"speaker": "Agent",    "text": rng.choice(greetings)},
        {"speaker": "Customer", "text": f"Hi, my name is {customer_name} and my account number is {account_id}."},
        {"speaker": "Agent",    "text": f"Thank you, {customer_name}. I've pulled up your account. Can you verify the last four digits of the Social Security Number or PIN on file?"},
        {"speaker": "Customer", "text": "Sure, it's [VERIFIED]."},
        {"speaker": "Agent",    "text": "Perfect, I've got you verified. What can I help you with today?"},
    ]


def _closer_clean(agent_name: str, customer_name: str,
                  rng: np.random.Generator) -> List[Turn]:
    closes = [
        f"Is there anything else I can help you with today, {customer_name}?",
        f"Great. Is there anything else on your account I can take care of for you today?",
    ]
    return [
        {"speaker": "Agent",    "text": rng.choice(closes)},
        {"speaker": "Customer", "text": "No, that's all. Thank you so much."},
        {"speaker": "Agent",    "text": f"Wonderful! Thank you for being a NovaWireless customer, {customer_name}. Have a great day!"},
    ]


def _closer_frustrated(agent_name: str, customer_name: str,
                       rng: np.random.Generator) -> List[Turn]:
    return [
        {"speaker": "Agent",    "text": f"I understand your frustration, {customer_name}, and I'm sorry for the experience. Is there anything else I can assist with?"},
        {"speaker": "Customer", "text": "No. I just hope this is actually taken care of this time."},
        {"speaker": "Agent",    "text": "I completely understand. I've documented everything on the account. Thank you for your patience today."},
    ]


# ---------------------------------------------------------------------------
# Scenario body builders
# ---------------------------------------------------------------------------

def _body_clean_billing(customer: dict, credit_info: dict,
                        rng: np.random.Generator) -> List[Turn]:
    charge = round(customer.get("monthly_charges", 85.0), 2)
    applied = credit_info.get("credit_applied", False)
    amount  = credit_info.get("credit_amount", 15.0)

    credit_line = (
        f"Let me check your account history. Since you've been with us for a while and this is your "
        f"first time calling about a charge like this, I'm going to go ahead and apply a one-time "
        f"courtesy credit of ${amount:.2f} to your account. That will post within one to two billing cycles."
        if applied else
        "Let me look into this further. I've documented the charge on your account and submitted a review request. "
        "You'll receive follow-up within two business days."
    )
    customer_response = (
        "That's great, thank you. I really appreciate that."
        if applied else
        "Okay, I hope it gets sorted out."
    )
    return [
        {"speaker": "Customer", "text": f"I'm calling about my bill. It's higher than I expected this month — it shows ${charge + 15:.2f} but I thought it was supposed to be ${charge:.2f}."},
        {"speaker": "Agent",    "text": "Let me take a look at your billing details right now. One moment please."},
        {"speaker": "Agent",    "text": f"I can see what happened here. You were charged a one-time activation fee of $15.00 that posted this cycle. That's a standard fee when a new service feature was added. I do see your base plan remains at ${charge:.2f}."},
        {"speaker": "Customer", "text": "Oh, okay. I wasn't expecting that. Can that be waived?"},
        {"speaker": "Agent",    "text": credit_line},
        {"speaker": "Customer", "text": customer_response},
        {"speaker": "Agent",    "text": "Absolutely, happy to help. Your account is in great standing and we appreciate your loyalty."},
    ]


def _body_clean_network(customer: dict, credit_info: dict,
                        rng: np.random.Generator) -> List[Turn]:
    # Network calls don't get credits — credit_info will always be none here
    return [
        {"speaker": "Customer", "text": "I've been having dropped calls at my house for the past week. It's really frustrating."},
        {"speaker": "Agent",    "text": "I'm sorry to hear that. Let me pull up the network status for your area and also check your device's signal profile."},
        {"speaker": "Agent",    "text": "I can see there was scheduled maintenance in your area that completed two days ago. I'm also sending a network refresh signal to your device right now. Can you turn your phone off and back on while we're on the call?"},
        {"speaker": "Customer", "text": "Sure, give me a second... Okay it's back on."},
        {"speaker": "Agent",    "text": "Perfect. Your device has re-registered to the network. The maintenance should have resolved the underlying issue. If you continue to see dropped calls after 24 hours, please call us back and we'll escalate this to our network engineering team with a ticket already opened."},
        {"speaker": "Customer", "text": "Okay, that makes sense. I'll keep an eye on it."},
    ]


def _body_clean_device(customer: dict, credit_info: dict,
                       rng: np.random.Generator) -> List[Turn]:
    return [
        {"speaker": "Customer", "text": "My phone won't activate. I just got it and it keeps saying SIM not provisioned."},
        {"speaker": "Agent",    "text": "I can definitely help with that. Let me verify the IMEI on the device against what's showing on your account."},
        {"speaker": "Agent",    "text": "I see the issue — the IMEI registered during the order doesn't match what the network is seeing. This sometimes happens during fulfillment. I'm going to update the IMEI on your account right now to match your device. This will take about two minutes to propagate."},
        {"speaker": "Customer", "text": "Okay, should I restart my phone?"},
        {"speaker": "Agent",    "text": "Yes, restart after about two minutes and you should be good to go. I'm also adding a note to your account in case you need to reference this call."},
        {"speaker": "Customer", "text": "Perfect. Thank you, that was easy."},
    ]


def _body_clean_promo(customer: dict, credit_info: dict,
                      rng: np.random.Generator) -> List[Turn]:
    applied = credit_info.get("credit_applied", False)
    amount  = credit_info.get("credit_amount", 10.0)

    credit_line = (
        f"I see your autopay was enrolled but the discount wasn't triggered on the system side. "
        f"I've gone ahead and manually applied it. You'll see a ${amount:.2f} credit this cycle "
        f"and it will continue automatically going forward. I'm also documenting this so if there's "
        f"ever a question it's on record."
        if applied else
        "I see your autopay was enrolled. Let me submit a ticket to have the discount reviewed — "
        "it should be resolved within seven to ten business days."
    )
    return [
        {"speaker": "Customer", "text": "I was told I'd get a $10 per month discount for enrolling in autopay but I don't see it on my bill."},
        {"speaker": "Agent",    "text": "Let me look into that right away. I want to make sure that credit is applied correctly."},
        {"speaker": "Agent",    "text": credit_line},
        {"speaker": "Customer", "text": "Great, that's what I expected. Thank you for fixing that."},
        {"speaker": "Agent",    "text": "Of course. And just so you know, I verified you're fully qualified for this discount — it's applied correctly and you won't need to call about it again."},
    ]


def _body_clean_account_security(customer: dict, credit_info: dict,
                                  rng: np.random.Generator) -> List[Turn]:
    concerns = [
        "I got a text saying my SIM card was changed and I didn't authorize that.",
        "I got an email that someone logged into my account from a location I don't recognize.",
        "I think someone may have accessed my account. I'm seeing changes I didn't make.",
    ]
    customer_concern = concerns[int(rng.integers(0, len(concerns)))]
    return [
        {"speaker": "Customer", "text": customer_concern},
        {"speaker": "Agent",    "text": "I take that very seriously and I want to help you secure your account right now. I'm going to place a temporary security hold while we review recent activity. Can you confirm you have access to your account PIN?"},
        {"speaker": "Customer", "text": "Yes, I do."},
        {"speaker": "Agent",    "text": "Good. I'm reviewing your recent account activity now — login history and any changes made in the last 30 days."},
        {"speaker": "Agent",    "text": "I'm walking through your account activity right now. Let me confirm with you what was and wasn't authorized."},
        {"speaker": "Customer", "text": "Okay, please."},
        {"speaker": "Agent",    "text": "Everything I'm reviewing appears consistent with your normal usage pattern. As a precaution I'm going to reset your account security credentials, update your PIN, and flag this account for a 30-day security watch. You'll get a notification if anything unusual comes through."},
        {"speaker": "Customer", "text": "That makes me feel better. Thank you for taking it seriously."},
        {"speaker": "Agent",    "text": "Absolutely. Your account security is our priority. I've documented this interaction and I'll give you a case number for your records."},
    ]


def _body_unresolvable(customer: dict, credit_info: dict,
                       rng: np.random.Generator) -> List[Turn]:
    applied = credit_info.get("credit_applied", False)
    amount  = credit_info.get("credit_amount", 25.0)

    credit_line = (
        f"Here's what I can do for you: I'm going to escalate this to our port dispute team, "
        f"which has direct communication channels with other carriers. I'm also applying a "
        f"${amount:.2f} service credit to your account for the inconvenience. The dispute team "
        f"will contact you within 48 business hours with an update."
        if applied else
        "Here's what I can do for you: I'm going to escalate this to our port dispute team, "
        "which has direct communication channels with other carriers. They will contact you "
        "within 48 business hours with an update."
    )
    return [
        {"speaker": "Customer", "text": "I've been trying to get my number ported from my old carrier for three weeks. Every time I call they say it's in progress but nothing happens."},
        {"speaker": "Agent",    "text": "I completely understand how frustrating that is and I want to help you get to the bottom of this. Let me pull up the porting status right now."},
        {"speaker": "Agent",    "text": "I can see the port request is still showing as pending on our end. The delay is actually coming from your previous carrier — they have an internal hold on the number. There's a regulatory process called a port freeze that can cause this, and unfortunately I'm not able to override what your previous carrier has placed on the account."},
        {"speaker": "Customer", "text": "So what am I supposed to do? I've already paid for service here."},
        {"speaker": "Agent",    "text": credit_line},
        {"speaker": "Customer", "text": "Okay. I'm frustrated but I understand it's not your fault. I appreciate you being straight with me."},
        {"speaker": "Agent",    "text": "I appreciate your patience. I wish I could fix it right now — I want to be honest with you that I can't, but the escalation team absolutely can. You'll hear from them."},
    ]


def _body_gamed_metric(customer: dict, scenario_meta: dict, credit_info: dict,
                       rng: np.random.Generator) -> List[Turn]:
    """Rep uses a credit as hush money to close the call and protect FCR metric."""
    aware   = scenario_meta.get("rep_aware_gaming", False)
    applied = credit_info.get("credit_applied", False)
    amount  = credit_info.get("credit_amount", 15.0)

    if aware and applied:
        # Rep knowingly applies a bandaid credit to suppress callback
        return [
            {"speaker": "Customer", "text": "My bill went up by $30 this month and nobody told me it was going to change. I want an explanation."},
            {"speaker": "Agent",    "text": "I completely hear you, and I'm really sorry about that confusion. Let me see what happened here."},
            {"speaker": "Agent",    "text": "So what I'm seeing is a rate adjustment that went through — these are sometimes applied to bring plans in line with current pricing. I know that's not easy to hear."},
            {"speaker": "Customer", "text": "Can you reverse it? I didn't agree to this."},
            {"speaker": "Agent",    "text": f"I understand. What I can do is apply a one-time courtesy credit of ${amount:.2f} to soften the impact this cycle. The rate itself is a system change I'm not able to modify, but this credit should help."},
            {"speaker": "Customer", "text": "That's not the same as fixing it. But fine, whatever."},
            {"speaker": "Agent",    "text": "I hear you. Is there anything else I can help with today? I want to make sure we close this out in a way that works for you."},
            {"speaker": "Customer", "text": "No. I guess not. But I'm not happy about it."},
            {"speaker": "Agent",    "text": "I completely understand. Thank you for bearing with me. I've noted your concerns on the account."},
        ]
    elif aware and not applied:
        return [
            {"speaker": "Customer", "text": "My bill went up by $30 this month and nobody told me it was going to change. I want an explanation."},
            {"speaker": "Agent",    "text": "I completely hear you, and I'm really sorry about that confusion. Let me see what happened here."},
            {"speaker": "Agent",    "text": "So what I'm seeing is a rate adjustment that went through — these are sometimes applied to bring plans in line with current pricing. I know that's not easy to hear."},
            {"speaker": "Customer", "text": "Can you reverse it? I didn't agree to this."},
            {"speaker": "Agent",    "text": "Unfortunately the rate itself is a system change I'm not able to modify from my end. I've documented your concern and submitted a review request. Someone should follow up with you within five to seven business days."},
            {"speaker": "Customer", "text": "That's not good enough. But fine."},
            {"speaker": "Agent",    "text": "I understand. Thank you for your patience. Is there anything else I can help with today?"},
            {"speaker": "Customer", "text": "No."},
        ]
    else:
        # Rep doesn't check eligibility, just deflects
        return [
            {"speaker": "Customer", "text": "I have a promotion I was told I'd get when I signed up and it's never shown on any of my bills."},
            {"speaker": "Agent",    "text": "I'm sorry about that. Let me look into the promotion for you."},
            {"speaker": "Agent",    "text": "I see a note on the account about a promotional offer. It looks like it may not have been fully set up on the system side. Let me make a note here and submit a ticket to have that reviewed."},
            {"speaker": "Customer", "text": "When will that be resolved? I've been waiting months."},
            {"speaker": "Agent",    "text": "Typically these tickets are reviewed within seven to ten business days. I'll make sure the notes are detailed so they have everything they need."},
            {"speaker": "Customer", "text": "Fine. I hope it actually gets done this time."},
            {"speaker": "Agent",    "text": "I understand. Is there anything else I can help you with today?"},
            {"speaker": "Customer", "text": "No."},
        ]


def _body_fraud_store_promo(customer: dict, scenario_meta: dict, credit_info: dict,
                             rng: np.random.Generator) -> List[Turn]:
    aware   = scenario_meta.get("rep_aware_gaming", False)
    applied = credit_info.get("credit_applied", False)
    amount  = credit_info.get("credit_amount", 25.0)

    closing_agent_line = (
        f"Your frustration is completely valid. I've escalated this as a store promise dispute "
        f"and flagged the new line for review. Someone from store accountability will be in touch "
        f"within 3-5 business days. I'm applying a ${amount:.2f} credit to your account in the "
        f"meantime for the inconvenience."
        if applied else
        "Your frustration is completely valid. I've escalated this as a store promise dispute "
        "and flagged the new line for review. Someone from store accountability will be in touch "
        "within 3-5 business days."
    )
    return [
        {"speaker": "Customer", "text": "I went into a store last week and the rep told me I'd get $400 off my new phone if I added a line. I did it, but now my account shows none of that discount."},
        {"speaker": "Agent",    "text": "I want to help you with this. Let me pull up your account and look at what was added."},
        {"speaker": "Agent",    "text": "I can see the new line was added and a device was activated. However, I'm not seeing a $400 promotion attached to this account in our system."},
        {"speaker": "Customer", "text": "The rep in the store showed me on a tablet. They said it was a trade-in promotion. I have a text message they sent me about it."},
        {"speaker": "Agent",    "text": "I believe you, and I want to be transparent with you — the promotion you're describing requires a trade-in of a qualifying device, and I don't see a trade-in recorded on this order. " + ("I'm going to document everything you've told me and escalate this to our store relations team. In the meantime I can't apply a $400 credit I'm not authorized for, but this will be investigated." if aware else "Let me check to see if there's anything I can do here... I don't see a matching promotion in our current catalog that I can apply, but let me submit a research ticket and have this reviewed.")},
        {"speaker": "Customer", "text": "This is ridiculous. The store rep basically lied to me. What am I supposed to do with a line I don't need?"},
        {"speaker": "Agent",    "text": closing_agent_line},
        {"speaker": "Customer", "text": "Fine. But if this isn't fixed I'm cancelling everything."},
        {"speaker": "Agent",    "text": "I hear you. I've documented that as well. Thank you for bringing this to us."},
    ]


def _body_fraud_line_add(customer: dict, scenario_meta: dict, credit_info: dict,
                          rng: np.random.Generator) -> List[Turn]:
    lines   = customer.get("lines_on_account", 2)
    applied = credit_info.get("credit_applied", False)
    amount  = credit_info.get("credit_amount", 40.0)

    credit_line = (
        f"I'm flagging this account right now for an unauthorized line investigation. I'm also "
        f"applying a ${amount:.2f} credit to cover this month's charge on that line while the "
        f"investigation is open. The dispute team will contact you within 48 hours."
        if applied else
        "I'm flagging this account right now for an unauthorized line investigation. "
        "The dispute team will contact you within 48 hours."
    )
    return [
        {"speaker": "Customer", "text": f"I got my bill and there's a charge for a line I never asked for. I went into the store to get a new phone, not to add a line. Now my bill is way higher than it should be."},
        {"speaker": "Agent",    "text": "I'm so sorry to hear that. Let me pull up your account right now and review what was set up."},
        {"speaker": "Agent",    "text": f"I can see you currently have {lines} lines on the account. Let me look at the line that was added recently and the associated equipment installment plan."},
        {"speaker": "Customer", "text": "I don't know anything about an installment plan either. I thought I just bought the phone outright."},
        {"speaker": "Agent",    "text": "I understand. Looking at the line that was added — there is a device installment plan attached to it. This is important because per our policy, I cannot cancel a line that has an active EIP on it as a line that was never installed. I have to escalate this to our store dispute team to review the original transaction."},
        {"speaker": "Customer", "text": "So you're telling me I'm stuck paying for a line I didn't want because the store put a payment plan on it?"},
        {"speaker": "Agent",    "text": f"I completely understand why that's upsetting, and I want to be honest with you — yes, there's a process we have to go through to dispute this properly. {credit_line}"},
        {"speaker": "Customer", "text": "I can't believe this. I'm going to file a complaint."},
        {"speaker": "Agent",    "text": "You have every right to. I've documented your intent to file a complaint and provided you the case number for this investigation. I'm truly sorry this happened."},
    ]


def _body_fraud_hic_exchange(customer: dict, scenario_meta: dict, credit_info: dict,
                              rng: np.random.Generator) -> List[Turn]:
    applied = credit_info.get("credit_applied", False)
    amount  = credit_info.get("credit_amount", 50.0)

    waiver_line = (
        f"No. I'm waiving the non-return fee right now — that's ${amount:.2f} removed from your "
        f"account, documented. I'm also escalating this to our home internet line correction team "
        f"to merge these lines back and close the incorrectly opened one. This will take two to "
        f"three business days to fully process on the back end. I'm giving you a case number to reference."
        if applied else
        "No. I'm escalating this to our home internet line correction team to review the NRF and "
        "merge these lines back. This will take two to three business days. I'm giving you a case number."
    )
    return [
        {"speaker": "Customer", "text": "I got a bill showing a charge for not returning equipment and I have no idea what they're talking about. I went to the store because my home internet device wasn't working and they gave me a new one."},
        {"speaker": "Agent",    "text": "I'm so sorry about that. Let me pull this up. I want to understand exactly what happened on your account."},
        {"speaker": "Agent",    "text": "I can see here that a new line of service was added for home internet, and the previous line has a non-return fee applied to it. It looks like rather than processing an exchange on your existing line, the store opened a new line."},
        {"speaker": "Customer", "text": "That doesn't make any sense. I went in for a replacement device because mine was broken. I didn't want a new line. They didn't tell me anything about that."},
        {"speaker": "Agent",    "text": "I believe you completely. What should have happened is a warranty exchange on your existing home internet line. What happened instead created a second line, and when the original line showed no device returned, the system generated that non-return fee automatically."},
        {"speaker": "Customer", "text": "So I have to pay for a mistake the store made?"},
        {"speaker": "Agent",    "text": waiver_line},
        {"speaker": "Customer", "text": "Thank you. I was so confused about this charge. I thought they were trying to bill me for equipment I returned."},
        {"speaker": "Agent",    "text": "That's exactly what happened in effect, and it was not your error at all. I've made sure that's clearly documented."},
    ]


def _body_fraud_care_promo(customer: dict, scenario_meta: dict, credit_info: dict,
                            rng: np.random.Generator) -> List[Turn]:
    # No credit applied on care promo — it's a promise dispute, handled by override team
    aware = scenario_meta.get("rep_aware_gaming", False)
    return [
        {"speaker": "Customer", "text": "I called last week and a rep told me I qualified for a $20 per month discount on my plan if I agreed to keep my service for six more months. I agreed. But now I see nothing on my account about it."},
        {"speaker": "Agent",    "text": "I'm pulling up your account now and checking for any saved promotions or notes from your last interaction."},
        {"speaker": "Agent",    "text": "I see the note from the previous call indicating that offer was discussed. " + ("Let me verify whether your account actually qualifies for this promotion... I do want to be honest with you — reviewing the eligibility criteria, your account doesn't meet all the requirements for this specific discount. I'm not sure why it was offered. I'm going to flag this as a promised promotion and submit it for override review, because if it was promised to you it should be honored." if aware else "I see the note but I'm not finding the promotion code in our current offers. This may have been a limited time offer. Let me submit a research ticket to have this reviewed by our promotions team.")},
        {"speaker": "Customer", "text": "I literally agreed to stay because of that offer. If it's not going to happen I want to cancel."},
        {"speaker": "Agent",    "text": "I completely understand. I'm documenting this as a promised promotion dispute right now. Our promotions override team will review this within 48 hours. If the offer was promised, it will be honored retroactively. I don't want to lose you as a customer and I want to make sure you're treated fairly."},
        {"speaker": "Customer", "text": "Okay. But if nothing happens in 48 hours I'm calling back to cancel."},
        {"speaker": "Agent",    "text": "Noted. I've put that in the record. Thank you for giving us the chance to make it right."},
    ]


def _body_activation_clean(customer: dict, credit_info: dict,
                            rng: np.random.Generator) -> List[Turn]:
    device_options = ["iPhone", "Samsung Galaxy", "Google Pixel", "Motorola"]
    device = device_options[int(rng.integers(0, len(device_options)))]
    return [
        {"speaker": "Customer", "text": f"Hi, I just got a new {device} and I'm trying to activate it but I'm not sure what steps to take."},
        {"speaker": "Agent",    "text": "I can absolutely help you activate your new device. I'm going to walk you through the whole process. Do you have the SIM card that came with the phone, or is this an eSIM activation?"},
        {"speaker": "Customer", "text": "It came with a physical SIM card."},
        {"speaker": "Agent",    "text": "Perfect. Go ahead and insert the SIM card into the tray on the side of the phone, power it on, and let me know when you see the NovaWireless signal bars appear at the top."},
        {"speaker": "Customer", "text": "Okay, I put it in... I see one bar right now."},
        {"speaker": "Agent",    "text": "That's normal while it's registering. I'm sending a provisioning signal to the network right now to link your IMEI to your account. Give it about 60 seconds."},
        {"speaker": "Customer", "text": "Okay... it went to three bars. And now I'm seeing a NovaWireless network name."},
        {"speaker": "Agent",    "text": "That's exactly what we want to see. I'm confirming on my end that your device is now fully activated and registered to your account. Try making a quick test call or sending a text if you'd like to verify."},
        {"speaker": "Customer", "text": "I just texted my husband and it went through. This was much easier than I expected."},
        {"speaker": "Agent",    "text": "Wonderful! Your new device is fully active. I've added a note to your account with today's activation details. Is there anything else you'd like help setting up?"},
    ]


def _body_activation_failed(customer: dict, credit_info: dict,
                             rng: np.random.Generator) -> List[Turn]:
    error_options = [
        "SIM not provisioned",
        "device not recognized on the network",
        "IMEI flagged as incompatible",
    ]
    error   = error_options[int(rng.integers(0, len(error_options)))]
    applied = credit_info.get("credit_applied", False)
    amount  = credit_info.get("credit_amount", 7.50)

    credit_line = (
        f"I know that's incredibly frustrating and I'm sorry. Here's what I'm doing right now: "
        f"I'm opening a priority escalation ticket to our device provisioning team. They have "
        f"access to the backend tools I don't have. They typically resolve these within 4 to 24 "
        f"hours and you'll get a text confirmation when it's cleared. I'm also applying a "
        f"${amount:.2f} service credit to your account for the inconvenience."
        if applied else
        "I know that's incredibly frustrating and I'm sorry. Here's what I'm doing right now: "
        "I'm opening a priority escalation ticket to our device provisioning team. They typically "
        "resolve these within 4 to 24 hours and you'll get a text confirmation when it's cleared."
    )
    return [
        {"speaker": "Customer", "text": f"I'm trying to activate my new phone and it keeps showing an error — it says '{error}.' I've restarted it three times."},
        {"speaker": "Agent",    "text": "I'm sorry to hear that. Let me pull up your account and look at what the system is showing on our end for this device."},
        {"speaker": "Agent",    "text": "I can see the activation request was submitted but it's erroring out on our network side. Let me try re-provisioning the SIM remotely."},
        {"speaker": "Customer", "text": "Okay, I restarted again and it's still showing the same error."},
        {"speaker": "Agent",    "text": "I was afraid of that. The remote provisioning attempt isn't taking. This is a system-level issue on our end — I can see a provisioning error code that requires our technical backend team to manually clear. I cannot resolve this from my system right now."},
        {"speaker": "Customer", "text": "So what does that mean? I can't use my new phone?"},
        {"speaker": "Agent",    "text": credit_line},
        {"speaker": "Customer", "text": "I need my phone for work. Is there anything else that can be done?"},
        {"speaker": "Agent",    "text": "If it's urgent, you can take the device to a NovaWireless retail store — their on-site technicians have direct provisioning tools. I'd recommend calling ahead to confirm availability. Otherwise the ticket I just opened is the fastest remote path. I'll give you the ticket number right now."},
        {"speaker": "Customer", "text": "Okay. Give me the ticket number and I'll decide what to do."},
        {"speaker": "Agent",    "text": "Your ticket number is NOV-ACT-" + str(int(rng.integers(100000, 999999))) + ". I've also sent a confirmation to your email on file. I'm truly sorry we couldn't get this resolved on the call today."},
    ]


def _body_line_add_legitimate(customer: dict, credit_info: dict,
                               rng: np.random.Generator) -> List[Turn]:
    lines         = int(customer.get("lines_on_account", 1))
    charge        = round(float(customer.get("monthly_charges", 85.0)), 2)
    new_line_cost = 30.0
    return [
        {"speaker": "Customer", "text": "Hi, I'd like to add a new line to my account for my daughter. She's starting college and needs her own phone."},
        {"speaker": "Agent",    "text": "Congratulations — that's exciting! I'd be happy to help you add a line. Let me pull up your account so we can look at your current plan and the best options for adding a line."},
        {"speaker": "Agent",    "text": f"I can see you currently have {lines} line{'s' if lines > 1 else ''} on your account. Our current add-a-line promotion gives you an additional line for ${new_line_cost:.2f} per month, which includes unlimited talk and text and shared data on your current plan. Does that work for you?"},
        {"speaker": "Customer", "text": f"That sounds reasonable. So my bill would go from ${charge:.2f} to ${charge + new_line_cost:.2f}?"},
        {"speaker": "Agent",    "text": f"Exactly right. ${charge + new_line_cost:.2f} total going forward. I'll need a few details to set up the new line — will your daughter be bringing her own device or would she need one?"},
        {"speaker": "Customer", "text": "She has a phone already, she just needs the service."},
        {"speaker": "Agent",    "text": "Perfect — a bring-your-own-device activation. I'll set up the new line now. I'm going to send a SIM kit to the address on your account, which typically arrives in two to three business days. Once she receives it, she can call in or use the app to complete activation."},
        {"speaker": "Customer", "text": "That works. Will I see the charge on my next bill?"},
        {"speaker": "Agent",    "text": "Yes — you'll see a prorated charge for the remainder of this billing cycle plus the full amount for next month. I've documented everything on your account and you'll receive a confirmation email within a few minutes. Your account now shows the new line as pending activation."},
        {"speaker": "Customer", "text": "Perfect. Thank you for making this so easy."},
        {"speaker": "Agent",    "text": "Of course! We're happy to have your daughter joining NovaWireless. Is there anything else I can help you with today?"},
    ]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_transcript(
    scenario: str,
    call_type: str,
    agent: dict,
    customer: dict,
    scenario_meta: dict,
    credit_info: dict,
    rng: np.random.Generator,
) -> List[Turn]:
    """
    Returns a list of Turn dicts representing the full conversation.
    credit_info must be provided — it carries credit_applied, credit_amount,
    credit_type, and credit_authorized so the transcript matches the metadata exactly.
    """
    agent_name    = agent.get("rep_name", "Agent")
    customer_name = str(customer.get("customer_id", "Customer"))
    account_id    = customer.get("account_id", "ACCT-UNKNOWN")

    turns = _opener(agent_name, customer_name, account_id, rng)

    if scenario == "clean":
        ct = call_type
        if ct in ("Billing Dispute", "Payment Arrangement"):
            turns += _body_clean_billing(customer, credit_info, rng)
        elif ct == "Network Coverage":
            turns += _body_clean_network(customer, credit_info, rng)
        elif ct == "Device Issue":
            turns += _body_clean_device(customer, credit_info, rng)
        elif ct == "Promotion Inquiry":
            turns += _body_clean_promo(customer, credit_info, rng)
        elif ct in ("Account Inquiry", "Account Security"):
            turns += _body_clean_account_security(customer, credit_info, rng)
        elif ct == "International/Roaming":
            turns += _body_clean_network(customer, credit_info, rng)
        else:
            turns += _body_clean_billing(customer, credit_info, rng)
        turns += _closer_clean(agent_name, customer_name, rng)

    elif scenario == "unresolvable_clean":
        turns += _body_unresolvable(customer, credit_info, rng)
        turns += _closer_frustrated(agent_name, customer_name, rng)

    elif scenario == "gamed_metric":
        turns += _body_gamed_metric(customer, scenario_meta, credit_info, rng)
        turns += _closer_clean(agent_name, customer_name, rng)

    elif scenario == "fraud_store_promo":
        turns += _body_fraud_store_promo(customer, scenario_meta, credit_info, rng)
        turns += _closer_frustrated(agent_name, customer_name, rng)

    elif scenario == "fraud_line_add":
        turns += _body_fraud_line_add(customer, scenario_meta, credit_info, rng)
        turns += _closer_frustrated(agent_name, customer_name, rng)

    elif scenario == "fraud_hic_exchange":
        turns += _body_fraud_hic_exchange(customer, scenario_meta, credit_info, rng)
        turns += _closer_clean(agent_name, customer_name, rng)

    elif scenario == "fraud_care_promo":
        turns += _body_fraud_care_promo(customer, scenario_meta, credit_info, rng)
        turns += _closer_frustrated(agent_name, customer_name, rng)

    elif scenario == "activation_clean":
        turns += _body_activation_clean(customer, credit_info, rng)
        turns += _closer_clean(agent_name, customer_name, rng)

    elif scenario == "activation_failed":
        turns += _body_activation_failed(customer, credit_info, rng)
        turns += _closer_frustrated(agent_name, customer_name, rng)

    elif scenario == "line_add_legitimate":
        turns += _body_line_add_legitimate(customer, credit_info, rng)
        turns += _closer_clean(agent_name, customer_name, rng)

    return turns


def transcript_to_text(turns: List[Turn]) -> str:
    """Flatten turn list to readable string."""
    return "\n".join(f"[{t['speaker']}]: {t['text']}" for t in turns)
