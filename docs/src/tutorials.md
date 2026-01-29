---
outline: deep
---
# Examples & Tutorials

Explore the capabilities of Corleone.jl through these guided examples.

<script setup>
import { ref, computed } from 'vue'

const examples = [{"icon":"🎣","title":"The Lotka Volterra Fishing Problem","tags":["Lotka Volterra","Optimal Control"],"link":"./examples/the_lotka_volterra_fishing_problem","description":"A beginner-friendly guide to your first simulation."},{"icon":"🎣🎣","title":"The Lotka Volterra Fishing Problem 2","tags":["Lotka Volterra","Optimal Control","Control"],"link":"./examples/the_lotka_volterra_fishing_problem_2","description":"A beginner-friendly guide to your first simulation."}]
const searchQuery = ref('')

// Automatically extract unique tags from the data
const allTags = computed(() => {
  const tags = new Set()
  examples.forEach(ex => {
    if (ex.tags) ex.tags.forEach(t => tags.add(t))
  })
  return Array.from(tags).sort()
})

const setFilter = (tag) => {
  if (searchQuery.value === tag) {
    searchQuery.value = ''
  } else {
    searchQuery.value = tag
  }
}

const filteredExamples = computed(() => {
  const query = searchQuery.value.toLowerCase().trim()
  if (!query) return examples

  return examples.filter(ex => {
    const inTitle = ex.title.toLowerCase().includes(query)
    const inDesc = ex.description.toLowerCase().includes(query)
    const inTags = ex.tags.some(tag => tag.toLowerCase().includes(query))
    return inTitle || inDesc || inTags
  })
})
</script>

<div class="search-wrapper">
  <div class="input-group">
    <input 
      v-model="searchQuery" 
      type="text" 
      placeholder="Search by name, description, or tag..." 
      class="search-input"
    />
    <button v-if="searchQuery" @click="searchQuery = ''" class="clear-btn">✕</button>
  </div>

  <div class="tag-filter-bar">
    <span 
      class="tag filter-tag" 
      :class="{ 'active-tag': searchQuery === '' }"
      @click="searchQuery = ''"
    >
      All
    </span>
    <span 
      v-for="tag in allTags" 
      :key="tag" 
      class="tag filter-tag"
      :class="{ 'active-tag': searchQuery.toLowerCase() === tag.toLowerCase() }"
      @click="setFilter(tag)"
    >
      {{ tag }}
    </span>
  </div>
  
  <p class="search-stats">
    Showing {{ filteredExamples.length }} of {{ examples.length }} tutorials
  </p>
</div>

<div class="tutorial-grid">
  <a 
    v-for="ex in filteredExamples" 
    :key="ex.title" 
    :href="ex.link" 
    class="tutorial-card"
  >
    <div class="card-content">
      <div class="card-icon">{{ ex.icon }}</div>
      <div class="card-title">{{ ex.title }}</div>
      <div class="card-description">{{ ex.description }}</div>
      <div class="card-tags">
        <span 
          v-for="tag in ex.tags" 
          :key="tag" 
          class="tag clickable-tag"
          :class="{ 'active-tag': searchQuery.toLowerCase() === tag.toLowerCase() }"
          @click.stop.prevent="setFilter(tag)"
        >
          {{ tag }}
        </span>
      </div>
    </div>
  </a>
</div>

<div v-if="filteredExamples.length === 0" class="no-results">
  <p>No tutorials match "<strong>{{ searchQuery }}</strong>"</p>
  <button @click="searchQuery = ''" class="reset-link">Clear all filters</button>
</div>

<style>
.search-wrapper { margin: 2rem 0; }
.input-group { position: relative; display: flex; align-items: center; }

.search-input {
  width: 100%;
  padding: 12px 16px;
  padding-right: 40px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  transition: all 0.2s;
}

.tag-filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1rem;
}

.filter-tag {
  cursor: pointer;
  transition: transform 0.1s;
}

.filter-tag:hover {
  background-color: var(--vp-c-brand-soft);
}

.search-stats {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  margin-top: 1rem;
}

.tutorial-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.tutorial-card {
  text-decoration: none !important;
  background-color: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 1.5rem;
  transition: transform 0.25s, border-color 0.25s;
  display: flex;
  flex-direction: column;
  color: var(--vp-c-text-1) !important;
}

.tutorial-card:hover {
  transform: translateY(-4px);
  border-color: var(--vp-c-brand-1);
}

.tag {
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 10px;
  border-radius: 12px;
  text-transform: uppercase;
  background-color: var(--vp-c-default-soft);
  color: var(--vp-c-text-2);
  transition: all 0.2s;
}

.active-tag {
  background-color: var(--vp-c-brand-1) !important;
  color: white !important;
}

.clear-btn {
  position: absolute;
  right: 12px;
  background: none;
  border: none;
  color: var(--vp-c-text-2);
  cursor: pointer;
}
</style>
