const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FULL = path.join(ROOT, 'data', 'external', 'official', 'cards_full.json');
const OUT = path.join(ROOT, 'registry', 'card_functional_classification.json');
const REVIEWED_IDS_PATH = path.join(ROOT, 'registry', 'card_functional_reviewed_220_ids.json');
const MANUAL_OVERRIDES_PATH = path.join(ROOT, 'registry', 'card_functional_manual_overrides.json');

const TAX = new Set([
  'main_attacker',
  'tech_attacker',
  'attack_copy',
  'damage_mod',
  'snipe_spread',
  'win_condition',
  'basic_energy',
  'special_energy',
  'energy_accel',
  'draw',
  'tutor',
  'cycle',
  'consistency',
  'hand_disruption',
  'board_disruption',
  'energy_disruption',
  'tool_disruption',
  'stadium_disruption',
  'special_condition',
  'ability_disable',
  'stall_lock',
  'tanky',
  'wall',
  'protection',
  'heal',
  'retrieval',
  'bench_setup',
  'ability_engine',
  'on_ko_trigger',
  'coin_flip',
  'gust',
  'switch_pivot',
  'mill',
  'basic_mon',
  'evolution',
  'ex_mon',
  'mega_mon',
  'tera_mon',
  'stadium',
  'tool',
  'ace_spec',
]);

const CT = {
  0: 'Pokemon',
  1: 'Item',
  2: 'Tool',
  3: 'Supporter',
  4: 'Stadium',
  5: 'Basic Energy',
  6: 'Special Energy',
};

function lower(s) {
  return String(s || '').toLowerCase();
}

function words(s) {
  return String(s || '')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean);
}

function trimWhy(s) {
  const w = words(s);
  if (w.length <= 10) return s.trim();
  return w.slice(0, 10).join(' ');
}

function has(str, re) {
  return re.test(str);
}

function addIf(arr, tag, whyBits, bit) {
  if (TAX.has(tag)) {
    arr.push(tag);
    if (bit && whyBits.length < 2) {
      whyBits.push(bit);
    }
  }
}

function dedupeSorted(a) {
  return [...new Set(a)].sort();
}

function readJsonFile(p, fallback) {
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch (err) {
    if (err.code !== 'ENOENT') {
      console.warn(`warning: failed to read ${p}: ${err.message}`);
    }
    return fallback;
  }
}

const REVIEWED_IDS = new Set((readJsonFile(REVIEWED_IDS_PATH, []) || []).map(String));
const MANUAL_OVERRIDES = readJsonFile(MANUAL_OVERRIDES_PATH, {});

function textFields(obj, keys) {
  return keys
    .map(k => obj && typeof obj[k] === 'string' ? obj[k] : '')
    .filter(Boolean)
    .join(' ');
}

function classifyCard(card) {
  const id = String(card.id || '');
  const name = String(card.n || '');
  const text = [
    ...(Array.isArray(card.skills) ? card.skills.map(s => textFields(s, ['n', 't'])) : []),
    ...(Array.isArray(card.atk) ? card.atk.map(a => textFields(a, ['n', 't', 'e', 'desc'])) : []),
  ].join(' ');
  const lowerText = lower(text);
  const lowerName = lower(name);
  const tags = [];
  const whyBits = [];

  const ct = Number(card.ct);
  const hp = Number(card.hp) || 0;
  const isBasic = !!card.basic;
  const isS1 = !!card.s1;
  const isS2 = !!card.s2;
  const isAceSpec = !!card.aceSpec;
  const isExMon = !!card.ex || !!card.mega || /\bex\b/.test(lowerName);

  const attacks = Array.isArray(card.atk) ? card.atk : [];
  const lowerAttackNames = lower(attacks.map(a => a.n || '').join(' '));
  const damages = attacks
    .map(a => Number(a.d))
    .filter(x => Number.isFinite(x) && x > 0);
  const maxDamage = damages.length ? Math.max(...damages) : 0;
  const textDamageValues = [...lowerText.matchAll(/(?:does|do) (\d+) damage/g)]
    .map(m => Number(m[1]))
    .filter(x => Number.isFinite(x) && x > 0);
  const textMaxDamage = textDamageValues.length ? Math.max(...textDamageValues) : 0;
  const damageBonusValues = [...lowerText.matchAll(/(?:does|do|uses do) (\d+) more damage/g)]
    .map(m => Number(m[1]))
    .filter(x => Number.isFinite(x) && x > 0);
  const maxDamageBonus = damageBonusValues.length ? Math.max(...damageBonusValues) : 0;
  const selfCounterDamageMatch = lowerText.match(/put up to (\d+) damage counters on this pok.mon[\s\S]*?this attack does (\d+) damage for each damage counter/i);
  const selfCounterMaxDamage = selfCounterDamageMatch ? Number(selfCounterDamageMatch[1]) * Number(selfCounterDamageMatch[2]) : 0;
  const coinDamageMatch = lowerText.match(/flip (\d+) coins?[^.]*does (\d+) damage for each heads/i);
  const coinMaxDamage = coinDamageMatch ? Number(coinDamageMatch[1]) * Number(coinDamageMatch[2]) : 0;
  const effectiveMaxDamage = Math.max(maxDamage, textMaxDamage, maxDamage + maxDamageBonus, selfCounterMaxDamage, coinMaxDamage);
  const hasDamage = effectiveMaxDamage > 0;
  const variableDamageMatch = lowerText.match(/(?:this attack )?does (\d+) (?:more )?damage for each/);
  const variableDamageBase = variableDamageMatch ? Number(variableDamageMatch[1]) : 0;
  const hasVariableDamage = !!variableDamageBase || has(lowerText, /damage for each|more damage for each|for each energy attached|for each .* on your bench|for each .* in play/i);
  const hasOpenEndedCoinDamage = has(lowerText, /flip a coin until you get tails|flip .* until .* tails/);
  const hasAttackCopy = has(lowerText, /use .* attacks? .* as this attack|choose .* attacks? and use it as this attack|choose an attack .* use it as this attack|can use any attack from|choose 1 of your benched .* attacks? and use it as this attack/i);

  let lowConfidence = false;

  // Base type tags
  if (ct === 0) {
    if (isBasic) addIf(tags, 'basic_mon', whyBits, 'Basic stage pokemon.');
    if (isS1 || isS2) addIf(tags, 'evolution', whyBits);
    if (isExMon) addIf(tags, 'ex_mon', whyBits, 'Pokemon ex rule-box card.');
    if (card.mega) addIf(tags, 'mega_mon', whyBits, 'Mega Pokemon ex card.');
    if (card.tera) {
      addIf(tags, 'tera_mon', whyBits, 'Tera Pokemon rule-box card.');
      addIf(tags, 'protection', whyBits, 'Tera Bench damage protection.');
    }
  }
  if (ct === 4) {
    addIf(tags, 'stadium', whyBits, 'Board-state modifier in play.');
  }
  if (ct === 2) {
    addIf(tags, 'tool', whyBits, 'Tool attached to active assets.');
  }
  if (ct === 5) {
    addIf(tags, 'basic_energy', whyBits, 'Basic resource card.');
  }
  if (ct === 6) {
    addIf(tags, 'special_energy', whyBits, 'Special Energy with text.');
  }
  if (isAceSpec) {
    addIf(tags, 'ace_spec', whyBits, 'Deckbuilding restriction / ACE SPEC.');
  }

  // attack role
  if (ct === 0) {
    if (hasDamage) {
      if (effectiveMaxDamage >= 120 || (hasVariableDamage && !hasOpenEndedCoinDamage && (variableDamageBase >= 40 || hp >= 180))) {
        addIf(tags, 'main_attacker', whyBits, `Strong attack pressure (${effectiveMaxDamage}).`);
      } else {
        addIf(tags, 'tech_attacker', whyBits, `Useful attacker role (${effectiveMaxDamage} damage).`);
      }
    } else if (hasVariableDamage && !hasOpenEndedCoinDamage && (variableDamageBase >= 40 || hp >= 180)) {
      addIf(tags, 'main_attacker', whyBits, 'Scales attack damage by board resources.');
    }

    if (hp >= 230 && effectiveMaxDamage <= 60 && !hasVariableDamage && !hasAttackCopy) {
      addIf(tags, 'wall', whyBits, 'Very high HP with minimal offense.');
    }
    if (hp >= 130 && effectiveMaxDamage > 60 && effectiveMaxDamage < 120) {
      addIf(tags, 'tanky', whyBits, `Durable (${hp} HP).`);
    }
    if (hp >= 130 && effectiveMaxDamage >= 120) {
      addIf(tags, 'tanky', whyBits, `Durable (${hp} HP).`);
    }
    if (hp >= 130 && hasVariableDamage) {
      addIf(tags, 'tanky', whyBits, `Durable (${hp} HP) scaling attacker.`);
    }
    if (hp >= 130 && !hasDamage && !hasVariableDamage) {
      addIf(tags, 'tanky', whyBits, `Durable (${hp} HP) support shell.`);
      lowConfidence = true;
    }
    if (hp >= 230 && effectiveMaxDamage > 60 && effectiveMaxDamage >= 120 && !hasDamage) {
      lowConfidence = true;
    }
  }

  // text-derived combat utility
  const hasDraw = has(lowerText, /(draws?\s+\d+|draw cards|draw until|draw\b)/);
  const hasShuffleHand = has(lowerText, /shuffle your hand|shuffle\s+hand/);
  const hasDiscard = has(lowerText, /discard/);
  const hasOpponentHand = has(lowerText, /opponent.?s hand|from your opponent.?s hand|opponent discards .* hand|cards from their hand at random/);
  const hasSearchDeck = has(lowerText, /search your deck/);
  const hasSearchBench = has(lowerText, /search your deck for[^.]*pok.mon[^.]*put[^.]*onto your bench|search your deck[^.]*put[^.]*onto your bench|search your deck[^.]*put[^.]*onto .* bench/);
  const hasDeckLook = has(lowerText, /look at the (top|bottom) \d+ cards? of your deck/);
  const hasPutIntoHand = has(lowerText, /put (up to )?(\d+|a|any number|them|it|that card|those cards|.* you find there).* into your hand|put .* from (your|their) discard pile into .* hand|put .* from (your|their) discard into .* hand/);
  const hasGustSwitch = has(lowerText, /switch in 1 of your opponent.?s benched pok.mon to the active spot|switch out your opponent.?s active pok.mon to the bench|switch your opponent.?s active pok.mon|switch opponent.?s active pok.mon/);
  const hasSwitchPivot = has(lowerText, /switch this pok.mon with 1 of your benched pok.mon|switch it with your active pok.mon|switch 1 of your benched .* pok.mon[^.]*with your active pok.mon|switch your active pok.mon with 1 of your benched pok.mon|switch your active team rocket.?s pok.mon with 1 of your benched|exchange your active pok.mon|switch your active.*with 1 of your benched|switch it with 1 of your .* in play|put 1 of your pok.mon and all attached cards into your hand|retreat cost .* less|has no retreat cost|have no retreat cost/);
  const hasDrawDiscardCycle = has(lowerText, /shuffle your hand into your deck|discard your hand and draw|discard.*and draw|shuffle.*then draw|put .* hand .* deck/);
  const hasAttachEnergy = has(lowerText, /attach [^.]*energy[^.]*(from your [^.]*hand|from your [^.]*deck|from your [^.]*discard)|attach [^.]*energy cards? from/i) &&
    !has(lowerText, /whenever (you|your opponent|they) attach|when (you|your opponent|they) attach .* from (your|their) hand/);
  const hasAttachFromTop = has(lowerText, /search your deck and attach|search your deck for .*energy.*attach|search your deck .* energy .* attach|attach .*basic energy from|attach a basic energy card .* find there|attach any number of .*energy|attach up to \d+ .*energy|look at .* deck .* attach .* energy/);
  const hasAttackSnipe = has(lowerText, /does \d+ damage to (one|1|2|3|each|all) of your opponent.?s (benched )?.*pok.mon|does \d+ damage to each of your opponent.?s pok.mon|choose .*opponent.?s pok.mon.*do \d+ damage to it|choose .*opponent.?s pok.mon.*put \d+ damage counters on each|put \d+ damage counters on (one|1|2|3|each|all)? ?of your opponent.?s .*pok.mon|put \d+ damage counters on your opponent.?s .*pok.mon|put .*damage counters on .*opponent.?s benched .*pok.mon|move up to \d+ damage counters from .* to .*opponent.?s pok.mon|knock out .*opponent.?s benched .*pok.mon|damage to each benched pok.mon/i);
  const hasWinCond = has(lowerText, /(extra prize|\btake\b .* prize|\btakes\b .* prize|\btake\b an extra|\btakes? \d+ (more|fewer) prize|can.?t take any prize|deck[ -]?out|win (the|this) game|deck is empty|defending pok.mon will be knocked out|opponent.?s active pok.mon is knocked out|will be knocked out|both active pok.mon are knocked out|knock out .*opponent.?s .*pok.mon)/);
  const hasDamageBoost = has(lowerText, /(do \d+ more damage|does \d+ more damage|takes \d+ more damage|do .* more damage|damage counters|put \d+ damage counters|gets \+\d+ hp|gets -\d+ hp|ダメージ.*\+|「\+\d+」)/);
  const hasAttackCostChange = has(lowerText, /attack costs? .* less|attacks? .* costs? .* less|attack costs? .* more|attacks? .* costs? .* more/);
  const hasSpecialCondition = has(lowerText, /(burned|confused|poisoned|asleep|paralyzed|special condition|マヒ|どく|やけど|こんらん|ねむり)/);

  if (has(lowerText, /flip (a|\d+) coins?|coin flip|heads or tails|tails or heads/)) {
    addIf(tags, 'coin_flip', whyBits, 'Resolution depends on coin.');
  }

  if (hasAttackCopy) {
    addIf(tags, 'attack_copy', whyBits, 'Uses or copies another attack.');
  }

  const koTriggerText = lowerText
    .replace(/if your opponent.?s active pok.mon has any special energy attached,? it is knocked out/g, '')
    .replace(/if your opponent.?s active pok.mon has exactly \d+ damage counters on it,? that pok.mon is knocked out/g, '')
    .replace(/if your opponent.?s active pok.mon is a basic pok.mon,? it is knocked out/g, '')
    .replace(/even if[^.()]*knocked out[^.()]*/g, '')
    .replace(/if you use this ability[^.()]*knocked out[^.()]*/g, '')
    .replace(/if heads[^.()]*knocked out[^.()]*/g, '')
    .replace(/if tails[^.()]*knocked out[^.()]*/g, '')
    .replace(/would be knocked out[^.()]*/g, '');
  if (has(koTriggerText, /when .*knocked out|if .* is knocked out|if .* becomes knocked out|knocked out by damage .* (take|takes|put|place|attach|draw|discard|search|can't take|can.t take)|knocked out.*prize/)) {
    addIf(tags, 'on_ko_trigger', whyBits, 'Effect triggers on KO event.');
  }

  if (has(lowerText, /search your deck for/)) {
    addIf(tags, 'tutor', whyBits, 'Search text reaches deck.');
  }
  if (has(lowerText, /山札/) && has(lowerText, /手札/)) {
    addIf(tags, 'tutor', whyBits, 'Searches deck into hand.');
  }

  if (hasSearchBench && !has(lowerText, /when you play this pok.mon from your hand onto your bench|when you play this pokemon from your hand onto your bench/)) {
    addIf(tags, 'bench_setup', whyBits, 'Puts Pokémon directly onto Bench.');
  }

  if (hasSearchDeck && hasSearchBench && !has(lowerText, /when you play this pok.mon from your hand onto your bench|when you play this pokemon from your hand onto your bench/)) {
    addIf(tags, 'tutor', whyBits, 'Explicit deck search for Bench setup.');
  }

  if (hasDeckLook && hasPutIntoHand) {
    addIf(tags, 'draw', whyBits, 'Looks at deck cards and converts selections to hand.');
    if (has(lowerText, /(pok.mon|energy|supporter|trainer|item|tool)/)) {
      addIf(tags, 'tutor', whyBits, 'Deck look finds named resource classes.');
    }
  }

  if (hasDeckLook && has(lowerText, /(onto your bench|put .* you find there onto your bench)/)) {
    addIf(tags, 'bench_setup', whyBits, 'Deck look puts a Pokémon directly onto Bench.');
    addIf(tags, 'tutor', whyBits, 'Deck look finds a specific Bench target.');
  }

  if (hasDeckLook && has(lowerText, /(put them back in any order|put .* in any order|shuffle them and put them on the bottom|put .* on the bottom of your deck)/)) {
    addIf(tags, 'consistency', whyBits, 'Filters or orders upcoming deck cards.');
    addIf(tags, 'cycle', whyBits, 'Cycles weak top-deck cards away.');
  }

  if (ct === 0 &&
      has(lowerText, /(once during your turn|once during your first turn|at the end of your turn|at the end of this turn|each turn)/) &&
      !has(lowerText, /when you play this pok.mon from your hand|when you play this pokemon from your hand/)) {
    addIf(tags, 'ability_engine', whyBits, 'Recurring turn-cycle effect.');
  }

  if (has(lowerText, /\bheal\b|heal.*damage|remove .* damage|recover \d+ damage|recover all damage|move up to \d+ damage counters from 1 of your pok.mon|move all damage counters from 1 of your benched/)) {
    addIf(tags, 'heal', whyBits, 'Repairs damage on your Pokémon.');
  }
  if (has(lowerText, /(retriev|from (your|their) discard pile into .* hand|from (your|their) discard pile into .* deck|from (your|their) discard into .* hand|from (your|their) discard into .* deck|put .* from (your|their) discard pile into .* hand|put .* from (your|their) discard pile into .* deck|shuffle .* from (your|their) discard pile into .* deck)/)) {
    addIf(tags, 'retrieval', whyBits, 'Returns resources from discard.');
  }
  if (has(lowerText, /into your hand instead of the discard pile|into your hand instead of .*discard/)) {
    addIf(tags, 'retrieval', whyBits, 'Preserves resources from discard.');
  }

  if (has(lowerText, /(prevent all effects of attacks|can.?t use abilities|cannot use abilities|have no abilities|has no abilities|prevent all effects.*attacks)/)) {
    addIf(tags, 'ability_disable', whyBits, 'Restricts opponent effects.');
  }

  if (hasSpecialCondition && has(lowerText, /(opponent|active pok.mon is now|active pokemon is now|burned|poisoned|confused|asleep|paralyzed|相手|マヒ|どく|やけど|こんらん|ねむり)/)) {
    addIf(tags, 'special_condition', whyBits, 'Applies or modifies Special Conditions.');
  }

  if (has(lowerText, /(discard (all |up to \d+ |a )?pok.mon tools?|pok.mon tools? attached[^.]*discard|pok.mon tools lose effect|tool scrapper)/)) {
    addIf(tags, 'tool_disruption', whyBits, 'Removes or disables Pokemon Tools.');
  }

  if (has(lowerText, /(discard (a |the )?stadium in play|discard that stadium|discard .*stadium card|discard .*stadium in play)/)) {
    addIf(tags, 'stadium_disruption', whyBits, 'Removes or disrupts Stadium cards.');
  }

  if (has(lowerText, /(prevent all damage|prevent damage|takes .* less damage|take .* less damage|do \d+ less damage|does \d+ less damage|reduce damage by|damage reduction|reduce.*damage|can't be affected by any special conditions|can.t be affected by any special conditions)/)) {
    addIf(tags, 'protection', whyBits, 'Reduces opponent damage output.');
  }

  if (has(lowerText, /(would be knocked out[^.]*not knocked out|remaining hp becomes)/)) {
    addIf(tags, 'protection', whyBits, 'Prevents a would-be Knock Out.');
  }

  if (has(lowerText, /(move an energy from (1 of )?your opponent|move an energy from your opponent.?s active|discard (a |an |all |\d+ )?(\{.\} )?energy (card )?(attached to|from) your opponent|discard [^.]*your opponent.?s [^.]*energy|all energy from your opponent|opponent[^.]*energy cards? to discard|your opponent[^.]*unable to attach|put an energy attached to [^.]*opponent[^.]* into their hand|opponent reveals[^.]*energy card[^.]*bottom of their deck|discard [^.]*special energy [^.]*opponent)/)) {
    addIf(tags, 'energy_disruption', whyBits, 'Interferes with opponent Energy flow.');
  }

  if (has(lowerText, /(draw cards?|draw a card|draws? \d+ cards|draw 3 cards|draw 2 cards|draw 5 cards|draw 4 cards)/)) {
    addIf(tags, 'draw', whyBits, 'Direct card-gain effect.');
  }

  if (ct !== 5 && ct !== 6) {
    const hasCycle = hasDrawDiscardCycle;
    if (hasCycle) {
      addIf(tags, 'cycle', whyBits, 'Cycles hand/board through discard.');
    }
  }

  if (ct !== 5 && ct !== 6 && has(lowerText, /(discard the top \d+ cards? of your deck|put a card from .* hand on top of .* deck)/)) {
    addIf(tags, 'cycle', whyBits, 'Moves cards through deck/discard zones.');
  }

  if (ct !== 5 && ct !== 6 && hasDiscard && hasOpponentHand) {
    addIf(tags, 'hand_disruption', whyBits, 'Manipulates hand contents.');
  }

  if (ct !== 5 && ct !== 6 && has(lowerText, /(opponent.?s hand[^.]*shuffle|shuffle[^.]*opponent.?s hand|opponent counts the cards in their hand|opponent reveals their hand[\s\S]*?you discard|your opponent reveals their hand[\s\S]*?you discard|opponent reveals their hand[\s\S]*?discard .*you find there|your opponent reveals their hand[\s\S]*?discard .*you find there|opponent chooses [^.]* cards? from their hand[^.]*shuffles|opponent reveals their hand[\s\S]*?put .* bottom of their deck|your opponent reveals their hand[\s\S]*?put .* bottom of their deck|random card from your opponent.?s hand|cards from their hand at random|put [^.]*opponent.?s hand[^.]*deck)/)) {
    addIf(tags, 'hand_disruption', whyBits, 'Forces hand disruption for both players.');
  }

  if (ct !== 5 && ct !== 6 && has(lowerText, /(each player shuffles their hand|your opponent discards cards from their hand|each player discards cards from their hand)/)) {
    addIf(tags, 'hand_disruption', whyBits, 'Forces hand-size or shuffle reset.');
  }

  if (has(lowerText, /(discard the top (card|\d+ cards?) of your opponent.?s deck|discard .*from the top of your opponent.?s deck|your opponent discards .* from (the top of )?their deck|deck[ -]?out|deck is empty)/)) {
    addIf(tags, 'mill', whyBits, 'Forces opponent deck exhaustion.');
  }

  if (has(lowerText, /(search.*card|search.*deck|search your deck for)/) &&
      has(lowerText, /(attach|bench|evolution|basic pokémon|pokemon)/)) {
    addIf(tags, 'tutor', whyBits, 'Looks up specific cards.');
  }

  if (hasAttachEnergy || hasAttachFromTop || has(lowerText, /(move up to \d+ energy from your(?! opponent)|move .* energy from (1 of )?your(?! opponent) .* to|move a basic energy from)/)) {
    addIf(tags, 'energy_accel', whyBits, 'Allows additional Energy attachment.');
  }

  if (ct === 6 && has(lowerText, /(provides 2 in any combination|provides .*2 energy|provides \{.\}\{.\}|provides .*only 2 energy|provides \{.\}\{.\}\{.\})/)) {
    addIf(tags, 'energy_accel', whyBits, 'Special Energy provides extra Energy value.');
  }

  if (ct === 6 && has(lowerText, /attach this card from your discard pile/)) {
    addIf(tags, 'energy_accel', whyBits, 'Special Energy recurs itself from discard.');
  }

  if (hasAttackCostChange) {
    if (has(lowerText, /less/)) {
      addIf(tags, 'energy_accel', whyBits, 'Reduces attack Energy requirements.');
    }
    if (has(lowerText, /more/)) {
      addIf(tags, 'stall_lock', whyBits, 'Raises attack cost pressure.');
    }
  }

  if (has(lowerText, /retreat cost .* more|can.?t retreat|cannot retreat|attack doesn.?t happen|attack does not happen/)) {
    addIf(tags, 'stall_lock', whyBits, 'Raises retreat cost pressure.');
  }

  if (has(lowerText, /(defending pok.mon can.t use attacks|defending pok.mon cannot use attacks|opponent.?s active pok.mon[^.]*can.t use attacks|opponent.?s active pok.mon[^.]*cannot use attacks|your opponent[^.]*can.t use attacks|your opponent[^.]*cannot use attacks)/)) {
    addIf(tags, 'stall_lock', whyBits, 'Prevents opponent attack use.');
  }

  if (has(lowerText, /(no item|can.?t play any item|cannot play any item|item lock|player may not play|can.?t use card from hand|can.?t play trainer|cannot play supporter|cannot play pokemon|can.?t play any .* cards? from their hand|can.?t play any pok.mon .* from their hand|can.?t play any pok.mon from their hand to evolve|cannot play any pok.mon from their hand to evolve|can.?t use that attack)/)) {
    addIf(tags, 'stall_lock', whyBits, 'Prevents key action classes.');
  }

  if (has(lowerText, /(paralyzed|asleep|confused|マヒ|こんらん|ねむり|can.?t retreat|cannot retreat|don.t recover|doesn.t recover)/) && has(lowerText, /(opponent|active pok.mon is now|active pokemon is now|相手のバトルポケモン|don.t recover|doesn.t recover)/)) {
    addIf(tags, 'stall_lock', whyBits, 'Applies or extends Special Condition pressure.');
  }

  if (hasAttackSnipe) {
    addIf(tags, 'snipe_spread', whyBits, 'Can target or damage bench space.');
  }
  if (hasGustSwitch) {
    addIf(tags, 'gust', whyBits, 'Forces opponent Active switch-in pressure.');
  }
  if (hasSwitchPivot) {
    addIf(tags, 'switch_pivot', whyBits, 'Provides a user-side active/bench pivot swap.');
  }

  if (has(lowerText, /(discard the defending pok.mon|discard your opponent.?s active pok.mon|shuffle [^.]*opponent.?s [^.]*pok.mon[^.]* into their deck|choose \d+ of your opponent.?s benched [\s\S]*?shuffle those pok.mon|choose 1 of your opponent.?s benched [\s\S]*?shuffle that pok.mon|devolve each of your opponent.?s evolved pok.mon|shuffling the highest stage evolution card[^.]*into your opponent.?s deck|opponent.?s benched [^.]* into their deck|shuffle all of your opponent.?s benched|put [^.]*opponent.?s [^.]*pok.mon[^.]* into their deck|put any number of basic pok.mon you find there onto their bench)/)) {
    addIf(tags, 'board_disruption', whyBits, 'Removes opponent board pieces.');
  }

  if (has(lowerText, /put [^.]* you find there onto your opponent.?s bench/)) {
    addIf(tags, 'board_disruption', whyBits, 'Moves opponent hand Pokemon directly onto their Bench.');
  }

  if (hasWinCond) {
    addIf(tags, 'win_condition', whyBits, 'Potential route to victory beyond chip damage.');
  }

  if (has(lowerText, /if your opponent.?s active pok.mon is a basic pok.mon,? it is knocked out/)) {
    addIf(tags, 'win_condition', whyBits, 'Immediately Knocks Out eligible opponent Active Pokemon.');
  }

  if (hasDamageBoost) {
    if (has(lowerText, /gets \+\d+ hp/)) {
      addIf(tags, 'tanky', whyBits, 'Raises HP durability.');
    } else if (ct !== 0 || has(lowerText, /attacks used by [^.]* do \d+ more damage|attacks used by [^.]* does \d+ more damage|takes \d+ more damage from attacks|put \d+ more damage counters|put \d+ damage counters on that pok.mon|put \d+ damage counters on the attacking pok.mon|put \d+ damage counters on each pok.mon|put damage counters on the attacking pok.mon equal to|move all damage counters from .* to your opponent.?s active|move any number of damage counters on your opponent.?s pok.mon|weakness [^.]* is now|ダメージ.*\+|「\+\d+」/)) {
      addIf(tags, 'damage_mod', whyBits, 'Boosts or places damage without being an attacker.');
    } else if (!tags.includes('main_attacker')) {
      addIf(tags, 'tech_attacker', whyBits, 'Modifies attack pressure or places damage.');
    }
  }

  if (hasSwitchPivot) {
    addIf(tags, 'switch_pivot', whyBits, 'Improves active/bench or retreat flexibility.');
  }

  if (has(lowerText, /(put [^.]* onto your bench|have up to \d+ pok.mon on their bench|can have up to \d+ pok.mon on .* bench)/) &&
      !has(lowerText, /when you play this pok.mon from your hand onto your bench|when you play this pokemon from your hand onto your bench/)) {
    addIf(tags, 'bench_setup', whyBits, 'Expands or fills Bench space.');
  }

  if (has(lowerText, /(evolve|evolves|devolve|evolution cards|skipping the stage 1|can evolve during)/) &&
      !has(lowerText, /opponent[^.]*can.?t|they can.?t play any pok.mon from their hand to evolve|cannot play any pok.mon from their hand to evolve/)) {
    addIf(tags, 'evolution', whyBits, 'Changes evolution timing or stage flow.');
    if (has(lowerText, /(search your deck|can evolve during|skipping the stage 1|put it onto this pok.mon to evolve|evolves from this pok.mon|evolution cards? .* hand)/)) {
      addIf(tags, 'consistency', whyBits, 'Smooths evolution access.');
    }
  }

  if (has(lowerText, /(discard[^.]*prize|extra prize|takes? \d+ (more|fewer) prize|can.?t take any prize)/)) {
    addIf(tags, 'win_condition', whyBits, 'Prize-shift / extra win math.');
  }

  if (!lowerText.trim() && ct !== 5 && ct !== 6 && !(ct === 0 && attacks.length)) {
    lowConfidence = true;
  }

  if (ct === 0 && !hasDamage && attacks.length && !tags.includes('main_attacker')) {
    addIf(tags, 'tech_attacker', whyBits, 'Effect or variable-damage attack is present.');
    if (has(lowerAttackNames, /(hide|guard|take it easy)/)) {
      addIf(tags, 'protection', whyBits, 'Attack name implies defensive utility.');
    }
  }


  if (has(lowerText, /((put|place) \d+ damage counters? on (1|each) of your opponent.?s pok.mon|move all damage counters from [^.]* to 1 of your opponent.?s pok.mon|double the number of damage counters on each of your opponent.?s pok.mon)/)) {
    addIf(tags, 'snipe_spread', whyBits, 'Damage-counter placement can hit Benched/any opponent Pokemon.');
  }

  if (has(lowerText, /(devolve [^.]*opponent|opponent.?s evolved pok.mon[^.]*highest stage evolution card[^.]*hand)/)) {
    addIf(tags, 'board_disruption', whyBits, 'Devolves or removes an opponent Evolution from board.');
  }

  if (has(lowerText, /opponent[^.]*shuffle their hand into their deck and draw \d+ cards/)) {
    addIf(tags, 'hand_disruption', whyBits, 'Shrinks or resets the opponent hand.');
  }

  if (has(lowerText, /takes \d+ less damage from attacks/)) {
    addIf(tags, 'protection', whyBits, 'Reduces incoming attack damage.');
  }

  const opponentOnlyDraw = has(lowerText, /opponent[^.]*shuffle their hand into their deck and draw \d+ cards/) && !has(lowerText, /(you (may )?draw|each player draws|each player draw|both players draw|draw cards until you)/);
  if (opponentOnlyDraw) {
    const drawIndex = tags.indexOf('draw');
    if (drawIndex >= 0) tags.splice(drawIndex, 1);
    if (tags.includes('consistency') && !tags.some(t => ['tutor', 'energy_accel', 'bench_setup', 'retrieval', 'cycle'].includes(t))) {
      const consistencyIndex = tags.indexOf('consistency');
      if (consistencyIndex >= 0) tags.splice(consistencyIndex, 1);
    }
  }

  if (tags.includes('consistency') && tags.includes('board_disruption') && !tags.some(t => ['draw', 'tutor', 'energy_accel', 'bench_setup', 'retrieval', 'cycle'].includes(t))) {
    const consistencyIndex = tags.indexOf('consistency');
    if (consistencyIndex >= 0) tags.splice(consistencyIndex, 1);
  }


  // Batch 667-755 corrections.
  if (ct === 0 && has(lowerText, /each basic \{.\} energy attached to all of your pok.mon provides \{.\}\{.\} energy/)) {
    addIf(tags, 'ability_engine', whyBits, 'Ongoing Ability changes Energy value.');
    addIf(tags, 'energy_accel', whyBits, 'Basic Energy provides extra Energy value.');
  }

  if (has(lowerText, /(if your opponent.?s active pok.mon has exactly \d+ damage counters[^.]*knocked out|choose a pok.mon in play[^.]*is knocked out)/)) {
    addIf(tags, 'win_condition', whyBits, 'Direct Knock Out effect beyond raw damage.');
  }

  if (has(lowerText, /basic \{.\} energy cards?[^.]*in your discard pile[\s\S]*?shuffle those cards into your deck/)) {
    addIf(tags, 'retrieval', whyBits, 'Recycles Basic Energy from discard into deck.');
  }

  if (has(lowerText, /(place|put) \d+ damage counters? on (that|your opponent.?s active|1 of your opponent.?s) pok.mon/)) {
    addIf(tags, 'damage_mod', whyBits, 'Places damage counters as pressure.');
  }

  if (has(lowerText, /whenever (they|your opponent) attach an energy card[^.]*place \d+ damage counters/)) {
    const accelIndex = tags.indexOf('energy_accel');
    if (accelIndex >= 0) tags.splice(accelIndex, 1);
    if (tags.includes('consistency') && !tags.some(t => ['draw', 'tutor', 'bench_setup', 'retrieval', 'cycle'].includes(t))) {
      const consistencyIndex = tags.indexOf('consistency');
      if (consistencyIndex >= 0) tags.splice(consistencyIndex, 1);
    }
  }

  if (tags.includes('cycle') && has(lowerText, /discard the top \d+ cards of your deck/) && !has(lowerText, /(draw|into your hand|shuffle your hand|discard your hand and draw)/)) {
    const cycleIndex = tags.indexOf('cycle');
    if (cycleIndex >= 0) tags.splice(cycleIndex, 1);
  }


  // Batch 756-839 corrections.
  if (ct === 0 && has(lowerText, /(each of your pok.mon[^.]*may have up to \d+ pok.mon tool|as long as this pok.mon[^.]*whenever your opponent)/)) {
    addIf(tags, 'ability_engine', whyBits, 'Ongoing Ability creates board value or disruption.');
  }

  if (tags.includes('tool_disruption') && has(lowerText, /if this ability goes away[^.]*discard pok.mon tools from those pok.mon until only 1 remains/)) {
    const toolIndex = tags.indexOf('tool_disruption');
    if (toolIndex >= 0) tags.splice(toolIndex, 1);
  }

  if (tags.includes('damage_mod') && has(lowerText, /if you attached energy to a pok.mon in this way, place \d+ damage counters on that pok.mon/)) {
    const damageModIndex = tags.indexOf('damage_mod');
    if (damageModIndex >= 0) tags.splice(damageModIndex, 1);
  }


  // Batch 840-921 corrections.
  if (has(lowerText, /does \d+ damage to each of \d+ of your opponent.?s pok.mon/)) {
    addIf(tags, 'snipe_spread', whyBits, 'Hits multiple chosen opponent Pokemon.');
  }

  if (has(lowerText, /(place|put) \d+ damage counters? on 1 of your opponent.?s benched pok.mon/)) {
    addIf(tags, 'snipe_spread', whyBits, 'Places counters on an opponent Benched Pokemon.');
    addIf(tags, 'damage_mod', whyBits, 'Places damage counters as pressure.');
  }

  if (has(lowerText, /switch this pok.mon with your active pok.mon/)) {
    addIf(tags, 'switch_pivot', whyBits, 'Switches with the Active Pokemon.');
  }

  if (has(lowerText, /lose any ability that requires/)) {
    addIf(tags, 'ability_disable', whyBits, 'Removes a class of Abilities.');
  }

  if (has(lowerText, /each player draws a card/)) {
    addIf(tags, 'draw', whyBits, 'Draws cards.');
    addIf(tags, 'consistency', whyBits, 'Improves access to cards.');
  }

  if (ct === 0 && has(lowerText, /whenever your opponent.?s active pok.mon moves to the bench/)) {
    addIf(tags, 'ability_engine', whyBits, 'Ongoing Ability punishes opponent movement.');
  }

  if (has(lowerText, /place \d+ damage counters on the attacking pok.mon/)) {
    addIf(tags, 'damage_mod', whyBits, 'Places damage counters as counter-pressure.');
  }


  // Batch 922-1003 corrections.
  if (has(lowerText, /prevent that damage/)) {
    addIf(tags, 'protection', whyBits, 'Prevents incoming damage.');
  }

  if (has(lowerText, /place damage counters on your opponent.?s active pok.mon until/)) {
    addIf(tags, 'damage_mod', whyBits, 'Places damage counters as pressure.');
  }

  if (has(lowerText, /if your opponent.?s active pok.mon has any special energy attached,? it is knocked out/)) {
    addIf(tags, 'win_condition', whyBits, 'Direct Knock Out effect beyond raw damage.');
  }

  if (tags.includes('stall_lock') && has(lowerText, /during your next turn, this pok.mon can.t retreat/) && !has(lowerText, /(defending pok.mon can.t retreat|opponent[^.]*can.t retreat|that pok.mon can.t retreat)/)) {
    const stallIndex = tags.indexOf('stall_lock');
    if (stallIndex >= 0) tags.splice(stallIndex, 1);
  }

  if (tags.includes('protection') && has(lowerAttackNames, /take it easy/) && !has(lowerText, /(prevent .*damage|prevent that damage|takes? .*less damage|would be knocked out|remaining hp|attacks used by [^.]* less damage)/)) {
    const protectionIndex = tags.indexOf('protection');
    if (protectionIndex >= 0) tags.splice(protectionIndex, 1);
  }


  // Batch 1004-1084 corrections.
  if (has(lowerText, /opponent shuffles their hand[\s\S]*?they draw \d+ cards/)) {
    addIf(tags, 'hand_disruption', whyBits, 'Resets opponent hand size.');
    const drawIndex = tags.indexOf('draw');
    if (drawIndex >= 0) tags.splice(drawIndex, 1);
    if (tags.includes('consistency') && !tags.some(t => ['tutor', 'energy_accel', 'bench_setup', 'retrieval', 'cycle'].includes(t))) {
      const consistencyIndex = tags.indexOf('consistency');
      if (consistencyIndex >= 0) tags.splice(consistencyIndex, 1);
    }
  }

  if (has(lowerText, /ignore all \{c\} energy in the costs of attacks used by this pok.mon/)) {
    addIf(tags, 'energy_accel', whyBits, 'Reduces attack Energy requirements.');
    addIf(tags, 'consistency', whyBits, 'Improves attack access.');
  }

  if (has(lowerText, /discard an energy from the attacking pok.mon/)) {
    addIf(tags, 'energy_disruption', whyBits, 'Removes Energy from the opponent attacker.');
  }

  if (has(lowerText, /choose basic \{.\} energy cards from your discard pile[\s\S]*?attach them/)) {
    addIf(tags, 'energy_accel', whyBits, 'Attaches Energy from discard.');
    addIf(tags, 'consistency', whyBits, 'Improves Energy access.');
  }

  if (has(lowerText, /prevent all effects of your opponent.?s pok.mon.?s abilities done to this pok.mon/)) {
    addIf(tags, 'ability_disable', whyBits, 'Prevents opponent Ability effects.');
  }

  if (has(lowerText, /for each of your opponent.?s pok.mon[\s\S]*?flip a coin[\s\S]*?if heads[\s\S]*?does \d+ damage to that pok.mon/)) {
    addIf(tags, 'snipe_spread', whyBits, 'Can damage multiple opponent Pokemon.');
  }


  // Batch 1085-1166 corrections.
  if (has(lowerText, /pok.mon \{ex\} in your discard pile[^.]*switch it with 1 of your pok.mon \{ex\} in play/)) {
    addIf(tags, 'retrieval', whyBits, 'Reuses a Pokemon from discard.');
    const damageModIndex = tags.indexOf('damage_mod');
    if (damageModIndex >= 0) tags.splice(damageModIndex, 1);
  }

  if (has(lowerText, /reveal the top \d+ cards of your opponent.?s deck[\s\S]*put those pok.mon onto their bench/)) {
    addIf(tags, 'board_disruption', whyBits, 'Forces opponent Bench changes from deck.');
    for (const tag of ['bench_setup', 'draw']) {
      const idx = tags.indexOf(tag);
      if (idx >= 0) tags.splice(idx, 1);
    }
  }

  if (has(lowerText, /can.t be put into your hand or deck from the discard pile/)) {
    const cycleIndex = tags.indexOf('cycle');
    if (cycleIndex >= 0) tags.splice(cycleIndex, 1);
  }

  if (has(lowerText, /discard an energy from 1 of your opponent.?s pok.mon/)) {
    addIf(tags, 'energy_disruption', whyBits, 'Removes opponent Energy.');
  }

  if (tags.includes('stall_lock') && has(lowerText, /(this card can.t retreat|this pok.mon can.t retreat)/) && !has(lowerText, /(attacks used by your opponent.?s basic pok.mon cost [^.]* more|defending pok.mon can.t retreat|that pok.mon can.t retreat|opponent[^.]*can.t retreat|retreat cost of both active|retreat cost [^.]* more)/)) {
    const stallIndex = tags.indexOf('stall_lock');
    if (stallIndex >= 0) tags.splice(stallIndex, 1);
  }

  if (has(lowerText, /move up to \d+ basic energy cards from that pok.mon to your benched pok.mon/)) {
    addIf(tags, 'energy_accel', whyBits, 'Preserves Energy by moving it to the Bench.');
  }

  if (has(lowerText, /move an energy from the attacking pok.mon to 1 of your opponent.?s benched pok.mon/)) {
    addIf(tags, 'energy_disruption', whyBits, 'Moves Energy off the opponent attacker.');
  }


  // Accompanying Flute cleanup.
  if (has(lowerText, /opponent.?s deck[\s\S]*onto their bench/)) {
    for (const tag of ['bench_setup', 'draw']) {
      const idx = tags.indexOf(tag);
      if (idx >= 0) tags.splice(idx, 1);
    }
    addIf(tags, 'board_disruption', whyBits, 'Forces opponent Bench changes from deck.');
  }


  // Accompanying Flute final duplicate cleanup.
  if (has(lowerText, /opponent.?s deck[\s\S]*onto their bench/)) {
    for (const tag of ['bench_setup', 'draw']) {
      let idx = tags.indexOf(tag);
      while (idx >= 0) {
        tags.splice(idx, 1);
        idx = tags.indexOf(tag);
      }
    }
    addIf(tags, 'board_disruption', whyBits, 'Forces opponent Bench changes from deck.');
  }

  // Avoid tagging drawback-style attacks as protection.
  if (tags.includes('protection') && has(lowerText, /this attack does \d+ less damage/) && !has(lowerText, /(prevent|damage done to|attacks used by|takes? [^.]*less damage|would be knocked out|remaining hp becomes|reduce)/)) {
    const protectionIndex = tags.indexOf('protection');
    if (protectionIndex >= 0) tags.splice(protectionIndex, 1);
    const whyIndex = whyBits.indexOf('Reduces opponent damage output.');
    if (whyIndex >= 0) whyBits.splice(whyIndex, 1);
  }

  if (ct === 0 && !hasDamage && !(tags.includes('tech_attacker') || tags.includes('main_attacker') || tags.includes('bench_setup') || tags.includes('ability_engine') || tags.includes('draw') || tags.includes('tutor') || tags.includes('hand_disruption') || tags.includes('win_condition') || tags.includes('heal') || tags.includes('protection') || tags.includes('ability_disable') || tags.includes('energy_disruption') || tags.includes('retrieval')) ) {
    lowConfidence = true;
  }

  if ((ct === 1 || ct === 3 || ct === 2) && tags.length === 0) {
    lowConfidence = true;
  }

  if (tags.includes('draw') || tags.includes('tutor') || tags.includes('energy_accel') || tags.includes('bench_setup') || tags.includes('retrieval')) {
    addIf(tags, 'consistency', whyBits, 'Smooths turn-to-turn consistency.');
  }

  if (ct === 2 && !tags.includes('tool') && hasDraw && has(lowerText, /attach/)) {
    addIf(tags, 'tool', whyBits, 'Tool slot attachment control.');
  }

  if (tags.length === 0) {
    if (ct === 5 || ct === 6) {
      // kept in base above
    }
  }

  if (tags.includes('tutor') || tags.includes('draw')) {
    addIf(tags, 'consistency', whyBits, 'Contributes flow consistency.');
  }

  const manual = MANUAL_OVERRIDES[id] || {};
  const manualTags = Array.isArray(manual.tags) ? manual.tags : [];
  const manualWhy = typeof manual.why === 'string' ? manual.why.trim() : '';
  const reviewed = REVIEWED_IDS.has(id);
  if (manualTags.length) {
    for (const t of manualTags) {
      if (TAX.has(t) && !tags.includes(t)) {
        tags.push(t);
      }
    }
    lowConfidence = false;
  }
  if (manualWhy) {
    whyBits.push(manualWhy);
  }
  if (reviewed) {
    lowConfidence = false;
  }

  if (!lowConfidence) {
    if (!whyBits.length) {
      whyBits.push('Role from static card type and HP.');
    }
  }

  if (lowConfidence) {
    whyBits.push('Low confidence from text ambiguity.');
  }


  // Final Accompanying Flute name cleanup.
  if (lowerName === 'accompanying flute') {
    for (const tag of ['bench_setup', 'draw', 'consistency']) {
      let idx = tags.indexOf(tag);
      while (idx >= 0) {
        tags.splice(idx, 1);
        idx = tags.indexOf(tag);
      }
    }
    addIf(tags, 'board_disruption', whyBits, 'Forces opponent Bench changes from deck.');
  }


  // Final Antique Fossil self-retreat cleanup.
  if (['antique cover fossil', 'antique plume fossil', 'antique jaw fossil', 'antique sail fossil'].includes(lowerName)) {
    let idx = tags.indexOf('stall_lock');
    while (idx >= 0) {
      tags.splice(idx, 1);
      idx = tags.indexOf('stall_lock');
    }
  }


  // Batch 1168-1249 corrections.
  if (tags.includes('stall_lock') && has(lowerText, /attacks used [^.]* cost [^.]* less/) && !has(lowerText, /(cost [^.]* more|retreat cost [^.]* more|can.t|cannot)/)) {
    let idx = tags.indexOf('stall_lock');
    while (idx >= 0) {
      tags.splice(idx, 1);
      idx = tags.indexOf('stall_lock');
    }
  }

  if (lowerName === 'core memory') {
    addIf(tags, 'main_attacker', whyBits, 'Tool grants a high-damage attack.');
  }

  if (has(lowerText, /search your deck for a card that has no abilities/)) {
    let idx = tags.indexOf('ability_disable');
    while (idx >= 0) {
      tags.splice(idx, 1);
      idx = tags.indexOf('ability_disable');
    }
  }

  if (lowerName === "grimsley's move") {
    let idx = tags.indexOf('cycle');
    while (idx >= 0) {
      tags.splice(idx, 1);
      idx = tags.indexOf('cycle');
    }
  }

  if (has(lowerText, /pok.mon tools attached to each pok.mon[^.]*have no effect/)) {
    addIf(tags, 'tool_disruption', whyBits, 'Disables Pokemon Tool effects.');
  }


  // Final Hop's Choice Band stall cleanup.
  if (lowerName.includes('hop') && lowerName.includes('choice band')) {
    let idx = tags.indexOf('stall_lock');
    while (idx >= 0) {
      tags.splice(idx, 1);
      idx = tags.indexOf('stall_lock');
    }
  }

  // Final stadium batch corrections from the manual full-set audit.
  if (has(lowerText, /discard[s]? pok.mon from (their|your) bench until (they|you) have \d+/) || has(lowerText, /both players discard pok.mon from their bench until they have \d+/)) {
    addIf(tags, 'board_disruption', whyBits, 'Can force Bench discards.');
  }

  if (has(lowerText, /(each player|that player) may search their deck for a basic pok.mon and put it onto their bench/)) {
    addIf(tags, 'bench_setup', whyBits, 'Searches Basic Pokemon directly onto Bench.');
    addIf(tags, 'tutor', whyBits, 'Searches deck for a Pokemon.');
    addIf(tags, 'consistency', whyBits, 'Improves setup access.');
  }

  if (has(lowerText, /can evolve .* during the turn they play/)) {
    addIf(tags, 'consistency', whyBits, 'Accelerates evolution setup.');
  }

  const unique = dedupeSorted(tags);

  if (!unique.length) {
    return {
      tags: [],
      why: 'No functional cues parsed clearly.'
    };
  }

  // Keep all parsed tags; downstream review needs complete structural + functional roles.
  const picked = unique;
  const why = trimWhy(picked.map((t, i) => {
    if (t === 'main_attacker' && effectiveMaxDamage >= 120) return `Main attacker (${effectiveMaxDamage} dmg).`;
    if (t === 'tech_attacker' && effectiveMaxDamage > 0) return `Support attacker (${effectiveMaxDamage} dmg).`;
    if (t === 'attack_copy') return 'Copies or reuses another attack.';
    if (t === 'damage_mod') return 'Damage boost or counter placement.';
    if (t === 'basic_energy') return 'Basic Energy card.';
    if (t === 'special_energy') return 'Special Energy with effect.';
    if (t === 'tool_disruption') return 'Pokemon Tool disruption.';
    if (t === 'stadium_disruption') return 'Stadium disruption.';
    if (t === 'special_condition') return 'Applies Special Conditions.';
    if (t === 'ex_mon') return 'Pokemon ex rule-box card.';
    if (t === 'mega_mon') return 'Mega Pokemon ex card.';
    if (t === 'tera_mon') return 'Tera Pokemon rule-box card.';
    if (t === 'draw') return 'Cards draw support.';
    if (t === 'tutor') return 'Card search support.';
    if (t === 'bench_setup') return 'Pokémon Bench setup aid.';
    if (t === 'energy_accel') return 'Extra energy attachment/flow.';
    if (t === 'consistency') return 'Reliability / consistency aid.';
    if (t === 'ability_engine') return 'Recurring engine-like ability.';
    return 'Observed competitive utility.';
  }).join(' ').slice(0, 200));

  const finalWhy = trimWhy(whyBits.slice(0, 2).join(' '));
  return {
    tags: picked,
    why: finalWhy || why,
  };
}

const data = JSON.parse(fs.readFileSync(FULL, 'utf8'));
const out = {};
for (const [id, card] of Object.entries(data)) {
  card.id = id;
  const cls = classifyCard(card);
  out[id] = {
    tags: cls.tags,
    why: cls.why,
  };
}

fs.writeFileSync(OUT, JSON.stringify(out, null, 0), 'utf8');
console.log(`wrote ${Object.keys(out).length} cards -> ${OUT}`);
